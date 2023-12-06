import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import librosa

"""
Customized Dataset creater of .wav audio files with labels {'Major': 0, 'Minor': 1}

Parameters:
    - data_folder(str): Path to the folder contains 'Major'(folder) and 'Minor'(folder).
    - transform(str, optional): If spesified, will apply transform. Defaults is None.
        - Has a STFT transform built-in, named 'STFT_transform'
        - Has a FFT transform built-in, named 'FFT_transform'
        - Has a spectrogram transform built-in, named 'Spectrogram_transform'

Returns:
    - class AudioDataset

Example:

    '''#With STFT transform
    from dataset_creater.py import *
    data_folder = "./data"
    dataset = AudioDataset(data_folder, transform=STFT_transform)
    '''

    '''#No transform
    from dataset_creater.py import *
    data_folder = "./data"
    dataset = AudioDataset(data_folder)
    '''

Remark:
    Automatically padding the waveform to identical length
"""

class AudioDataset(Dataset):
    def __init__(self, data_folder, transform=None, scale=(0, 1)):
        self.data_folder = data_folder
        self.transform = transform

        # Set class name and corresponding labels
        self.classes = ['Major', 'Minor']
        self.class_to_idx = {'Major': 0, 'Minor': 1}

        self.file_paths = []
        self.labels = []
        self.scale = scale

        # Explore all files in dataset folder and calculate max length
        self.max_length = 0
        for class_name in self.classes:
            class_folder = os.path.join(data_folder, class_name)
            if os.path.isdir(class_folder):
                class_idx = self.class_to_idx[class_name]
                files = [f for f in os.listdir(class_folder) if f.endswith('.wav')]
                for file_name in files:
                    file_path = os.path.join(class_folder, file_name)
                    waveform, _ = torchaudio.load(file_path)
                    length = waveform.size(1)
                    self.max_length = max(self.max_length, length)

                    self.file_paths.append(file_path)
                    self.labels.append(class_idx)

        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load the audio file
        waveform, _ = torchaudio.load(file_path)

        # Apply zero-padding if needed
        if waveform.size(1) < self.max_length:
            padding_size = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_size), 'constant', 0)

        # Apply transformations if needed
        if self.transform:
            waveform = self.transform(waveform)
        waveform = (waveform - self.scale[0]) / (self.scale[1] - self.scale[0])
        return waveform, label


# STFT transform to transform the audio file into frenquency domain
def STFT_transform(waveform, n_fft=400, hop_length=160):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
    
    # Convert the STFT to decibels
    stft_db = torchaudio.transforms.AmplitudeToDB()(stft)
    
    return stft_db

# FFT transform to transform the entire audio file into frequency domain
def FFT_transform(waveform):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = torch.fft.fft(waveform, dim = 1)

    return fft_result

def Spectrogram_transform(waveform, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute the spectrogram from an audio waveform using Librosa.

    Parameters:
    - waveform (torch.Tensor): Input audio waveform.
    - sr (int): Sample rate of the audio signal (default: 22050).
    - n_fft (int): Number of FFT points (default: 2048).
    - hop_length (int): Hop length between frames (default: 512).
    - n_mels (int): Number of Mel filterbanks (default: 128).

    Returns:
    - torch.Tensor: Spectrogram as a 2D tensor.
    """
    # Convert torch.Tensor to numpy array
    waveform_np = waveform.numpy()

    # Compute the spectrogram using Librosa
    spectrogram = librosa.feature.melspectrogram(
        y=waveform_np[0], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to decibels (log scale)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Convert to PyTorch tensor
    spectrogram_tensor = torch.from_numpy(spectrogram_db).unsqueeze(0).float()

    return spectrogram_tensor