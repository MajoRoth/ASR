# ASR
Several ASR implementations


## installaion
- Download the [AN4 dataset](https://drive.google.com/file/d/1MiPqJDm6gXayXZJ2LHeUbG0UNZfNagF/view?usp=sharing).
- run `pip install -r requirments.txt`
- for conda: run `conda env create -f environment.yml`
- experiment with the models!
## models
All available models are cartesian product of the following acoustic models and ctc decoders configurations.

### Acoustic Models

1. Linear Layer
2. LSTM
3. DeepSpeech Toy
4. DeepSpeech Small
5. DeepSpeech Large

### CTC Decoders
1. Greedy CTC
2. Lexicon CTC
3. LM CTC

you can evaluate all models using `evaluate.py`, or try a specific configuration with `try.py`


## Training
### Train command example:
`python train.py --conf confs/linear_acoustic.json --logger wandb`

## Evaluation
The following command enables you to evaluate and observer the wer of all available models on train, test and val data
### Eval command example:
`python evaluate.py --conf confs/linear_acoustic.json`

## Try
A CLI which enables you to construct the ASR model you want and try to transcribe with it :)
### Eval command example:
`python try.py --conf confs/archive/try.json`

## Benchmarks
Evaluation results on all data are detailed in the paper. we appended here the graphs on the validation data.

[graph](https://github.com/MajoRoth/ASR/blob/main/data/static/augmented_benchmarks.png)



