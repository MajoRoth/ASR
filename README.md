# ASR
Several ASR implementations


## installaion
- Download the [AN4 dataset](https://drive.google.com/file/d/1MiPqJDm6gXayXZJ2LHeUbG0UNZfNagF/view?usp=sharing).
- run `pip install -r requirments.txt`
- for conda: run `conda env create -f environment.yml`
- experiment with the models!
## models
Models are ranked from worst to best.

1. Acoustic Greedy CTC - uses WAV2VEC to get the characters of the recording, and then naively concatenates to produce a sentence. 
Achieves WER of 0.78 on the val data.


## Training
### Train command example:
`python train.py --conf confs/linear_acoustic.json --logger wandb`

## Evaluation
### Eval command example:
`python evaluate.py --conf confs/linear_acoustic.json`
