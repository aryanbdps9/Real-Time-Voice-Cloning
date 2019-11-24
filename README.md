# Any to english audio dubber
This repository is a fork of (this)[https://github.com/CorentinJ/Real-Time-Voice-Cloning] repository.

## How to run
First run `demo_cli.py` using the instructions given in (original repo)[https://github.com/CorentinJ/Real-Time-Voice-Cloning]. Then run:
`CUDA_VISIBLE_DEVICES=1 python3 audio_breaker.py path/to/source/audio.wav path/to/subtitles -nj 6 -name output_dir`
`output_dir` is the name of output directory. You will find it in your home directory The script will produce a file called `<output_dir>_out_gen.wav`.

Be careful! Source must be a `.wav` file