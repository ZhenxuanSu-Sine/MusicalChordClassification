# Musical Chord Classification

## Project Purpose
This project aims to implement machine learning or deep learning-based methods for classifying major and minor chords. The project includes a variety of methods, from non-machine learning baselines to advanced deep learning models, providing a comprehensive exploration of chord classification techniques.

## Repository Structure
The repository contains several key files:

- `dataset_creater.py`: This script is used to generate the dataset for the project. It processes audio files and applies various transforms for chord classification.

- `nonML_baseline.ipynb`: This Jupyter notebook contains the implementation of the non-machine learning baseline method for chord classification.

- `ML_Frequency_domain.ipynb`: This Jupyter notebook contains the implementation of a machine learning method for chord classification, which operates in the frequency domain.

- `DL_Spectrogram.ipynb`: This Jupyter notebook contains the implementation of a deep learning model for chord classification, which takes spectrograms as input.

- `DL_MFCC.ipynb`: This Jupyter notebook contains the implementation of a deep learning model for chord classification, which takes Mel-frequency cepstral coefficients (MFCCs) as input.

## Data
The data for this project is not included in the repository. It should be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification/data) and moved to the `./data` folder in the repository.

## Setup and Execution
To set up and execute the project, follow these steps:

1. Clone the repository to your local machine.
2. Download the data and move it to the `./data` folder in the repository.
3. Install the necessary dependencies (listed below).
5. (To train) Open and run one of the four the Jupyter notebooks to execute the chord classification methods.
5. (To load and deploy the model) Use `model.load_state_dict(torch.load(PATH_TO_FILE))`

## Dependencies
This project requires the following Python libraries:

- python=3.8.18
- numpy=1.24.3
- pytorch=2.1.1
- cuda=12.1
- scipy=1.10.1
- matplotlib=3.7.2
- scikit-learn=1.3.2

These can be installed using pip:

```
pip install numpy scipy matplotlib sklearn torch torchvision torchaudio
```

Or with conda:

```
conda install numpy scipy matplotlib scikit-learn pytorch torchvision torchaudio -c pytorch
```

Please ensure that you have the correct versions of these libraries installed before running the project.