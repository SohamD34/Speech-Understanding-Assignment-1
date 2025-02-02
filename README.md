# CSL7770 Speech Understanding Assignment-1 
## Submitted by - Soham Deshmukh (B21EE067)

This is the code repository for submission to the Assignment 1 of CSL7770: Speech Understanding Course.
It contains source code files, scripts and reports for the problem statements given in the assignment.

## Directory Structure
```
root
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
├── Question 1
│       └── Report_Question_1.pdf
└── Question 2
        ├── Report_Question_2.pdf
        ├── Task A
        │      ├── models/                      (created during runtime)
        │      ├── spectrograms/                (created during runtime)
        │      ├── audio_dataset.py
        │      ├── cnn.py
        │      ├── train.py
        │      ├── utils.py
        │      ├── windows.py
        │      └── script.ipynb
        │
        └── Task B
               ├── songs_data/                  (contains .mp3 files for songs)
               ├── utils.py
               └── script.ipynb   
```

## Directions for using the repository

First, clone the repository to your local machine and navigate to the repository using the following commands:
```
> cd <folder_name>
> git clone https://github.com/SohamD34/Speech-Understanding-Assignment-1.git
> cd Speech-Understanding-Assignment-1
```

Now we need to set up the Python (Conda) environment for this repository. Execute the following commands in the terminal.
```
> conda create -p b21ee067 -y python==3.10.12
> conda activate b21ee067/
> pip install -r requirements.txt
```

## Question 1
The report is located at ```../Question 1/Report_Question_1.pdf``` .

## Question 2
The report for both Part A and B is located at ```../Question 2/Report_Question_2.pdf``` .Please follow along the below instructions to run and check the codes for Question 2.

Navigate to the root folder 'Speech-Understanding-Assignment-1'. If already working in this directory, ignore this step.
```
> cd ../Speech-Understanding-Assignment-1/
```

### Task A
#### 1. Download the dataset.
In the command line, execute the following commands. <br />
```
> mkdir data
> cd data 
> wget https://goo.gl/8hY5ER
> tar -zxvf 8hY5ER
```
Make sure your ```data``` directory looks like this -
```
data
  └── UrbanSound8K
          ├── audio
          │      ├── fold1
          │      ├── fold2
          │      :
          │      └── fold10
          ├── metadata
          │      └── UrbanSound8K.csv
          ├── FREESOUNDCREDITS.txt
          └── UrbanSound8K_README.txt
```
#### 2. Navigate to the folder 'Task A'.
```
> cd ..
> cd Question 2/Task A/
```
This directory contains all files necessary to run Task A. The directory structure is given above.
* ```audio_dataset.py``` - contains implementations for custom PyTorch Dataset classes
* ```cnn.py``` - contains implementation of CNN architecture for classification
* ```train.py``` - contains training script function
* ```utils.py``` - contains implementations of additional helper functions
* ```windows.py``` - scratch implementations of windowing techniques
* ```script.ipynb``` - notebook where the entire pipeline is executed
#### 3. Run the script notebook.
```
> jupyter notebook script.ipynb
```
* You can run all the cells in the notebook serially. 
* You can observe the spectrograms in the ```spectrograms/``` directory that gets created during runtime. 
* You can also access the ```.pkl``` (for ML models) and ```.pth``` files (for PyTorch neural networks) in the ```models/``` directory that gets created during runtime.


### Task B
In the command line, navigate to the folder ```Task B``` using the following commands. <br />
```
> cd ../Task B/
```
The folder ```songs_data/``` contains 4 songs (.mp3 files) of different genres - 
* Agar Tum Saath Ho - Bollywood.mp3
* Counting Stars - Pop.mp3
* Laal Ishq - Semi-Classical.mp3
* Saadda Haq - Rock.mp3

Run the script notebook.
```
> jupyter notebook script.ipynb
```
You can run all the cells in the notebook serially and the spectrograms will be visible in the notebook itself. 
