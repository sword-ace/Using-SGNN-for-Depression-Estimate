# SGNN

# Schematizing Language to Context-level Depressive Features Using GNN with Schema Encoders
This repo contains scripts to model depression in text.

### Data
The data used can be downloaded from the [Distress Analysis Interview Corpus](http://dcapswoz.ict.usc.edu/), and contains audio, video, and text of interviews with 142 subjects, about 30% of whom had some level of depression.

### Note
The data including real human participants cannot be released in public. Data accessing permission is required before using of those data. More details can be found [here](http://dcapswoz.ict.usc.edu/).

### Files
The repo contains the following files:

- **sgnn.py** which contains the sgnn model
- **main.py**  which contains  the methods used to train the model and the evaluation of the model
- **requirements.txt** which are the libraries used in the conda environment of this project.
- **helper** which contains the method of converting the text to graph

## Results

Our model achieves the following performance on :

### PHQ regression

![results](res1.jpg)

<!--### Loss Curve-->
<!---->
<!--![loss](./resulta/2.jpg)-->

pyTorch back-end was used for modeling.

## Libraries
I used the following librarires:
```
Python version: 3.6
torch==1.5.0+cu101
cuda10.1
tensorflow-gpu=1.15.2=0
tensorflow-gpu-base=1.15.2=py36h01caf0a_0
```

