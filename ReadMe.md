# Re-Implementing CRAFT-Character Region Awareness for Text Detection


## Objective

- [X] Reproduce weak-supervision training as mentioned in the paper https://arxiv.org/pdf/1904.01941.pdf
- [ ] Generate character bbox on all the popular data sets.
- [ ] Expose pre-trained models with command line interface to synthesize results on custom images


## Clone the repository

    git clone https://github.com/autonise/CRAFT-Remade.git
    cd CRAFT-Remade

## Option 1: Conda Environment Installation
    conda env create -f environment.yml
    conda activate craft

## Option 2: Pip Installation
    pip install -r requirements.txt

## Running on custom images

Put the images inside a folder.
<br>Get a pre-trained model from the pre-trained model list (Currently only strong supervision using SYNTH-Text available)
<br>Run the command - 

    python main.py train_synth --mode=synthesize --model=./model/final_model.pkl --folder=./input
    
## Pre-trained models

### Strong Supervision

SynthText - https://drive.google.com/open?id=1qnLM_iMnR1P_6OLoUoFtrReHe4bpFW3T<br>
    
### Weak Supervision

- [ ] ICDAR 2013 - In Progress
- [ ] ICDAR 2015 - In Progress
- [ ] ICDAR 2017 - yet_to_be_completed
- [ ] Total Text - yet_to_be_completed
- [ ] MS-COCO - yet_to_be_completed
    
## Pre-generated on popular data sets

- [ ] ICDAR 2013 - In Progress
- [ ] ICDAR 2015 - In Progress
- [ ] ICDAR 2017 - yet_to_be_completed
- [ ] Total Text - yet_to_be_completed
- [ ] MS-COCO - yet_to_be_completed
    
## How to train on your own dataset

Download the pre-trained model on Synthetic dataset at https://drive.google.com/open?id=1qnLM_iMnR1P_6OLoUoFtrReHe4bpFW3T
<br> Make your own custom dataloader as in train_weak_supervision/dataloader.DataLoaderMIX
<br> Run the command - 
    
    python main.py weak_supervision --model=/path/to/pre-trained/Synth-Text-Model --iterations=epochs-of-weak-supervision
