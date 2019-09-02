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

    python main.py synthesize --model=./model/final_model.pkl --folder=./input
    
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
    
## How to train the model from scratch

### Strong Supervision on Synthetic dataset

Download the pre-trained model on Synthetic dataset at https://drive.google.com/open?id=1qnLM_iMnR1P_6OLoUoFtrReHe4bpFW3T
<br> Otherwise if you want to train from scratch
<br> Run the command - 
    
    python main.py train_synth
    
<br> To test your model on SynthText, Run the command -
    
    python main.py test_synth --model /path/to/model
    
### Weak Supervision

#### First Pre-Process your dataset

*Currently Supported - [IC13, IC15]

The assumed structure of the dataset is

    .
    ├── Generated (This folder will contain the weak-supervision intermediate targets)
    └── Images
        ├── test
        │   ├── img_1.jpg
        │   ├── img_2.jpg
        │   ├── img_3.jpg
        │   ├── img_4.jpg
        │   └── img_5.jpg
        │   └── ...
        ├── test_gt.json (This can be generated using the pre_process function described below)
        ├── train
        │   ├── img_1.jpg
        │   ├── img_2.jpg
        │   ├── img_3.jpg
        │   ├── img_4.jpg
        │   └── img_5.jpg
        │   └── ...
        └── train_gt.json (This can be generated using the pre_process function described below)

To generate the json files for IC13 - 

    In config.py change the corresponding values
    
    'ic13': {
		'train': {
			'target_json_path': None,  --> path to where you want the target json file (Images/train_gt.json)
			'target_folder_path': None,  --> path to where you downloaded the train gt (ch2_training_localization_transcription_gt)
		},
		'test': {
			'target_json_path': None,  --> path to where you want the target json file (Images/test_gt.json)
			'target_folder_path': None,  --> path to where you downloaded the train gt (Challenge2_Test_Task1_GT)
		}
		
	Run the command - 
	
	python main.py pre_process --dataset IC13
		
To generate the json files for IC15 - 

    In config.py change the corresponding values
    
    'ic15': {
		'train': {
			'target_json_path': None,  --> path to where you want the target json file (Images/train_gt.json)
			'target_folder_path': None,  --> path to where you downloaded the train gt (ch4_training_localization_transcription_gt)
		},
		'test': {
			'target_json_path': None,  --> path to where you want the target json file (Images/test_gt.json)
			'target_folder_path': None,  --> path to where you downloaded the train gt (Challenge4_Test_Task1_GT)
		}
		
	python main.py pre_process --dataset IC15

#### Second Train your model based on weak-supervision

<br> Run the command - 

    python main.py weak_supervision --model /path/to/strong/supervision/model --iterations <num_of_iterations(20)>
    
This will train the weak supervision model for the number of iterations you specified
