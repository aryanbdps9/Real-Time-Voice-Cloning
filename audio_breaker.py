from scipy.ndimage.morphology import binary_dilation
import encoder.params_data as eparams
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
from typing import Optional, Union, Dict, List
import numpy as np
import webrtcvad
import librosa
import struct
import srt
from sklearn.cluster import DBSCAN, KMeans
import argparse
import os
from os import path
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

def load(filename):
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    return

int16_max = (2 ** 15) - 1

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("audio_path")
parser.add_argument("srt_path")
parser.add_argument("-nj", "--n_jobs", type=int, default=6)
parser.add_argument("-tot_time", "--total_time_in_sec", type=int, default=4)
# parser.add_argument("-o", "--out", default="outs")
# parser.add_argument('-c', "--cache_address", default=None)
# parser.add_argument('-a', "--audio_cache_address", default=None)
parser.add_argument('-name', "--name", default=None)
parser.add_argument('--eps', type=float, default=0.001)
parser.add_argument('--min_samples', type=float, default=1)
parser.add_argument('--min_seconds', type=int, default=4)

parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default="encoder/saved_models/pretrained.pt",
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_dir", type=Path,
                    default="synthesizer/saved_models/logs-pretrained/",
                    help="Directory containing the synthesizer model")
parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default="vocoder/saved_models/pretrained/pretrained.pt",
                    help="Path to a saved vocoder")
parser.add_argument("--low_mem", action="store_true", help="If True, the memory used by the synthesizer will be freed after each use. Adds large "
                    "overhead but allows to save some GPU memory for lower-end GPUs.")
parser.add_argument("--no_sound", action="store_true",
                    help="If True, audio won't be played.")

args = parser.parse_args()
root = "~"
out_dir = root  + "/" + args.name

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


cache_address = out_dir + "/" + args.name + "_cache_address.pkl"
audio_cache_address = out_dir + "/" + args.name + "_audio_cache_address.pkl"
audios_cache = out_dir + "/" + args.name + "_audios_cache.pkl"
output = out_dir + "/" + args.name + "_out_gen.wav"

print_args(args, parser)
print("cache_address : ", cache_address)
print("audio_cache_address : ", audio_cache_address)
print("audios_cache : ", audios_cache)
print("output : ", output)

if not args.no_sound:
    import sounddevice as sd


print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
vocoder.load_model(args.voc_model_fpath)


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != eparams.sampling_rate:
        wav = librosa.resample(wav, source_sr, eparams.sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, eparams.audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        eparams.sampling_rate,
        n_fft=int(eparams.sampling_rate * eparams.mel_window_length / 1000),
        hop_length=int(eparams.sampling_rate * eparams.mel_window_step / 1000),
        n_mels=eparams.mel_n_channels
    )
    return frames.astype(np.float32).T

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (eparams.vad_window_length * eparams.sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=eparams.sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, eparams.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(eparams.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def audio_fpath2wav(audio_fpath: Union[str, Path]):
    # Read and normalise audio into nparr
    wav, source_sr = librosa.load(audio_fpath, sr=None)
    sr = source_sr
    ## Resample the wav if needed
    if source_sr is not None and source_sr != eparams.sampling_rate:
        wav = librosa.resample(wav, source_sr, eparams.sampling_rate)
        sr = eparams.sampling_rate
    ## Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(
        wav, eparams.audio_norm_target_dBFS, increase_only=True)
    return wav, sr

def break_file(audio_fpath: Union[str, Path], srt_fpath, cache_address=None):
    """
    It takes an audio file and corresponding sub and breaks it
    """
    print("entering break_file _single_version_...")
    # Read and normalise audio into nparr
    if (cache_address is not None and path.exists(cache_address)):
        return load(cache_address)
    wav, sr = audio_fpath2wav(audio_fpath)

    # read srt file
    with open(srt_fpath, 'r') as f:
        srt_str = f.read()
    subs = list(srt.parse(srt_str))
    audios = []
    for idx, sub in enumerate(subs):
        start_time  = sub.start.total_seconds()
        finish_time = sub.end.total_seconds()
        start_idx   = int(start_time*sr)
        finish_idx  = min(int(finish_time*sr)+1, wav.size)
        audio_ = {
            'idx'       : idx,
            'clip'      : wav[start_idx:finish_idx],
            'txt'       : sub.content,
            'start'     : start_time,
            'finish'    : finish_time,
            'samp_rate' : sr
        }
        audios.append(audio_)
        if (int(sub.end.total_seconds()*sr)+1 >= wav.size):
            break
    print("exiting break_file _single_version_...")
    return audios

def break_file2(audio_dpath: Union[str, Path], srt_fpath):
    print("entering break_file2...")
    # read srt file
    with open(srt_fpath, 'r') as f:
        srt_str = f.read()
    subs = list(srt.parse(srt_str))
    print("subs", len(subs))
    audios = []
    for idx, sub in enumerate(subs):
        sub_index = sub.index
        try:
            wav, sr = audio_fpath2wav(audio_fpath=os.path.join(audio_dpath, f"extract_{sub_index}.wav"))
        except Exception:
            continue
        start_time  = sub.start.total_seconds()
        finish_time = sub.end.total_seconds()
        start_idx   = int(start_time*sr)
        finish_idx  = min(int(finish_time*sr)+1, wav.size)
        audio_ = {
            'idx'       : idx,
            'clip'      : wav,
            'txt'       : sub.content,
            'start'     : start_time,
            'finish'    : finish_time,
            'samp_rate' : sr
        }
        audios.append(audio_)
        # original_fin_idx = int(sub.end.total_seconds()*sr)+1
        # if (original_fin_idx >= wav.size):
        #     print(f"[{idx}]: fin_idx: {original_fin_idx};\twav.size = {wav.size}", end="\t")
        #     print(f"start_time: {start_time};\tend_time:{finish_time};\tsr:{sr}")
        #     break
    return audios

def break_file3(audio_dpath: Union[str, Path], txt_fpath, total_time_in_sec, cache_address=None):
    print("entering break_file3...")

    if (cache_address is not None and path.exists(cache_address)):
        return load(cache_address)

    # read srt file
    with open(txt_fpath, 'r') as f:
        srt_str = f.readlines()

    texts = srt_str[0]
    audios = []
    # wav, sr = audio_fpath2wav(audio_fpath=os.path.join(audio_dpath, f".wav"))
    wav, sr = audio_fpath2wav(audio_fpath=os.path.join(audio_dpath))
    start_time = 0
    finish_time = total_time_in_sec
    start_idx   = int(start_time*sr)
    finish_idx  = min(int(finish_time*sr)+1, wav.size)
    audio_ = {
        'idx'       : 0,
        'clip'      : wav,
        'txt'       : texts,
        'start'     : start_time,
        'finish'    : finish_time,
        'samp_rate' : sr
    }
    audios.append(audio_)

    if cache_address is not None:
        save(audios, "../speech/" + txt_fpath + "_aud.pkl")
    
    return audios

def audio2dvec(audio_):
    clip = audio_['clip']
    return encoder.embed_utterance(clip).flatten()

def generate_labels(audios: List[Dict], n_jobs=6):
    print("entering generate_labels")
    # each vector is 256 dimensional
    dvecs = np.array([audio2dvec(audio_) for audio_ in audios])
    print("dvecs:\n", dvecs.shape)
    print("fwd pass to ")
    # db_ = KMeans(n_clusters=2)
    # db  = db_.fit(dvecs)
    # cluster_dist = np.linalg.norm(db.cluster_centers_[0] - db.cluster_centers_[1])
    # print("av.dist", cluster_dist, "\nnorms:", np.linalg.norm(db.cluster_centers_, axis=1))
    # av.dist was: 0.254


    # 0.2 -> -1; 0.4 -> -1[192], 0[296]; -1[12], 0[476]; 0.8 -> -1[1], 0[487] :: min_samples=5
    # 0.6 -> -1[8], 0[4], 1[476]; 0.7 (BEST) -> -1[4], 0[4], 1[480]; :: min_samples=4

    db_ = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='euclidean', n_jobs=n_jobs)
    labels  = db_.fit_predict(dvecs)
    print(labels)
    label_set = set(labels.tolist())
    print("labels:", label_set)
    means = {}
    label2idxs = {}
    for label in sorted(label_set):
        mask = (labels == label).reshape((-1,1))
        dvecs_masked = dvecs * mask
        sum_ = np.sum(dvecs_masked, axis=0)
        div  = np.sum(mask)
        means[label] = sum_/div if div > 0 else sum_
        indices = np.argwhere(labels == label).flatten()
        print("label:", label, "indices.shape", indices.shape)
        indices = indices.tolist()
        label2idxs[label] = indices
    return labels, means, label2idxs

def print_audios(audios, prop):
    if (prop == 'duration'):
        f = (lambda x: x['finish'] - x['start'])
    for idx, audio in enumerate(audios):
        print(idx)
        print(f(audio))

def audio_repeater(audio):
    duration = audio['finish'] - audio['start']
    num_repeats = int(np.ceil(args.min_seconds / duration))
    repeated = audio['txt'] * num_repeats
    return repeated, num_repeats

def translate_audios(audios, labels, means, cache_address=None):
    if (cache_address is not None and path.exists(cache_address)):
        savedict = load(cache_address)
        spec = savedict['spec']
        breaks = savedict['breaks']
        specs = savedict['specs']
        repeats = savedict['repeats']
        wav = savedict['wav']
        print("picking from cache:", savedict.keys())
        print("repeats:", repeats)
    else:
        extended_txts_nrepeats = [audio_repeater(audio) for audio in audios]
        print('extended_txts_nrepeats', len(extended_txts_nrepeats))
        repeats = [x[1] for x in extended_txts_nrepeats]
        embeds = [means[label] for label in labels]
        print('embeds', len(embeds))
        texts = [extended_txt_nrepeat[0] for extended_txt_nrepeat in extended_txts_nrepeats]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        print('specs ', len(specs))
        print('specs[0]', specs[0].shape)
        print("synthesis done")
        breaks = [spec.shape[1] for spec in specs]
        print('breaks', len(breaks))
        spec = np.concatenate(specs, axis=1)
        print("spec", spec.shape)
        wav = vocoder.infer_waveform(spec)
        print('wav (vocoder_out)', wav.shape)
        print("vocoding done!")

        if (cache_address is not None):
            print("putting into cache...")
            savedict = {
                'spec': spec,
                'breaks': breaks,
                'specs': specs,
                'repeats': repeats,
                'wav': wav
            }
            save(savedict, cache_address)
        librosa.output.write_wav('../wholewav.wav', wav.astype('float32'), synthesizer.sample_rate)
    # wav = 

    # b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    hop_size = 200
    b_ends = np.cumsum(np.array(breaks) * hop_size)
    # last = b_ends[-1]
    # b_ends = b_ends * wav.shape[0] / last
    # b_ends = b_ends.astype('int')

    b_starts = np.concatenate(([0], b_ends[:-1]))
    # b_ends = b_ends[1:]
    sample_rate=16000
    # sample_rate=8000
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    wavs = [wav[:wav.size // repeats[idx]] for idx, wav in enumerate(wavs)]
    wav = np.concatenate(wavs, axis=0)

    # wavs = [wav_item[0:wav_item.size // extended_txts_nrepeats[idx][1]] for idx, wav_item in enumerate(extended_txts_nrepeats)]
    # TODO remove repetitions
    # breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    # wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    return wav

def test_translation_v1(audio_path: Union[str, Path], srt_fpath, n_jobs=6, wav_fpath='out.wav', cache_address=None, audio_cache_address=None, audios_cache=None):
    print("entering test_generated_labels...")
    # if not os.path.exists(wav_fpath):
    #     os.makedirs(wav_fpath, exist_ok=True)
    if (audio_cache_address is not None and path.exists(audio_cache_address)):
        ddict = load(audio_cache_address)
        audios, labels, means, label2idxs = ddict['audios'], ddict['labels'], ddict['means'], ddict['l2i']
    else:
        if os.path.isfile(audio_path):
            audios = break_file(audio_path, srt_fpath, audios_cache)
        else:
            audios = break_file2(audio_path, srt_fpath)
        print(f"audios = {len(audios)}")
        labels, means, label2idxs = generate_labels(audios, n_jobs=6)
        ddict = {
            'audios': audios,
            'labels': labels,
            'means': means,
            'l2i': label2idxs
        }
        if (audio_cache_address is not None):
            save(ddict, audio_cache_address)

    print_audios(audios, 'duration')
    audios = audios[:60]
    labels = labels[:60]
    
    print('labels generated')
    wav = translate_audios(audios, labels, means, cache_address)
    librosa.output.write_wav(wav_fpath, wav.astype('float32'), synthesizer.sample_rate)

def test_translation_v2(audio_path: Union[str, Path], srt_fpath, n_jobs=6, total_time_in_sec=4, wav_fpath='myout.wav', cache_address=None, audio_cache_address=None, audios_cache=None):
    print("entering test_generated_labels...")
    # if not os.path.exists(wav_fpath):
    #     os.makedirs(wav_fpath, exist_ok=True)
    if (audio_cache_address is not None and path.exists(audio_cache_address)):
        ddict = load(audio_cache_address)
        audios, labels, means, label2idxs = ddict['audios'], ddict['labels'], ddict['means'], ddict['l2i']
    else:
        audios = break_file3(audio_path, srt_fpath, total_time_in_sec, audios_cache)
        print(f"audios = {len(audios)}")
        labels, means, label2idxs = generate_labels(audios, n_jobs=6)
        ddict = {
            'audios': audios,
            'labels': labels,
            'means': means,
            'l2i': label2idxs
        }
        if (audio_cache_address is not None):
            save(ddict, audio_cache_address)
    
    print_audios(audios, 'duration')
    audios = audios[:20]
    labels = labels[:20]
    
    print('labels generated')
    wav = translate_audios(audios, labels, means, cache_address)
    librosa.output.write_wav(wav_fpath, wav.astype('float32'), synthesizer.sample_rate)






# def test_generated_labels(audio_path: Union[str, Path], srt_fpath, n_jobs=6, store_path='outs'):
#     print("entering test_generated_labels...")
#     if not os.path.exists(store_path):
#         os.makedirs(store_path, exist_ok=True)
#     if os.path.isfile(audio_path):
#         audios = break_file(audio_path, srt_fpath)
#     else:
#         audios = break_file2(audio_path, srt_fpath)
#     print(f"audios = {len(audios)}")
#     labels, means, label2idxs = generate_labels(audios, n_jobs=6)

#     texts = []
#     embeds = []
#     for label in sorted(means):
#         label_idxs = label2idxs[label]
#         # print("label_idxs0,len:", type(label_idxs[label][0]), len(label_idxs))
#         # for l_idx in label_idxs[label]:
#         #     ali = audios[l_idx]
#         #     print(type(ali['txt']), end='\t')
#         transcript_l = [audios[l_idx]['txt'] for l_idx in label_idxs]
#         transcript = ' '.join(transcript_l)
#         texts.append(transcript)
#         embeds.append(means[label])
#     specs = synthesizer.synthesize_spectrograms(texts, embeds)
#     assert len(specs) == len(texts)
#     generated_wavs = [vocoder.infer_waveform(specs[i]) for i in range(len(specs))]

#     ## Post-generation
#     # There's a bug with sounddevice that makes the audio cut one second earlier, so we
#     # pad it.
#     generated_wavs = [np.pad(
#         generated_wav, (0, synthesizer.sample_rate), mode="constant") for generated_wav in generated_wavs]
#     for idx, generated_wav in enumerate(generated_wavs):
#         gen_fpath = os.path.join(store_path, f"generated_{idx}.wav")
#         librosa.output.write_wav(gen_fpath, generated_wav.astype('float32'),
#                                  synthesizer.sample_rate)
    
#     print("labels:", sorted(means))
#     first = sorted(means)[0]
#     print(f"type(label2idxs[{first}]) = ", type(label2idxs[first]))

#     for label in means:
#         # print("#"*20)
#         # print(f"label: {label}", type(label2idxs[label][0]))
#         wav_fpath = os.path.join(store_path, f"collected_{label}.wav")
#         label_wavs = [audios[idx]['clip'] for idx in label2idxs[label]]
#         label_wav = np.concatenate(label_wavs, axis=None)
#         librosa.output.write_wav(wav_fpath, label_wav.astype('float32'), synthesizer.sample_rate)
#     return labels, means, label2idxs


# test_generated_labels(args.audio_path, args.srt_path, n_jobs=args.n_jobs, store_path=args.out)
# python audio_breaker.py ../extracted_audio ../NamoSpeech.srt -o ../namogen

test_translation_v1(args.audio_path, args.srt_path, n_jobs=args.n_jobs, wav_fpath=output, cache_address=cache_address, audio_cache_address=audio_cache_address, audios_cache=audios_cache)
# python audio_breaker.py ..\Mitron.wav ..\Mitron.srt -o ..\MEout.wav -nj 6
# python audio_breaker.py /home/aryan/test_new.wav /home/aryan/part.srt -o /home/aryan/gen_test_new.wav -nj 6
# CUDA_VISIBLE_DEVICES=1 python audio_breaker.py /home/aryan/test_new.wav /home/aryan/part.srt -nj 6 -name "test_new"
# CUDA_VISIBLE_DEVICES=1 python audio_breaker.py ../Mitron.wav ../Mitron.srt -nj 6 -name "Mitron"

# test_translation_v2(args.audio_path, args.srt_path, n_jobs=args.n_jobs, total_time_in_sec=args.total_time_in_sec, cache_address=args.cache_address, audio_cache_address=args.audio_cache_address, audios_cache='ashish_audios.pkl')
# python audio_breaker.py ../speech/sample_aryan_hindi0.wav ../speech/sample_aryan_sub -nj 6 -tot_time 4 -c "my_cache_address.pkl" -a "my_audio_cache_address.pkl"