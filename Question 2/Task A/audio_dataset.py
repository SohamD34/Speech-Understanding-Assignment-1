import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class AudioDataset(Dataset):
    '''
    Class for loading raw audio files into tensor format
    '''
    def __init__(self, dataset, fixed_length):
        self.audio_data = dataset['X']
        self.labels = dataset['Y']
        self.label_mapping = {label: idx for idx, label in enumerate(set(self.labels))}
        self.fixed_length = fixed_length  

    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio, sr = self.audio_data[idx]
        audio = self._pad_or_truncate(audio)

        numeric_label = torch.tensor(self.label_mapping[self.labels[idx]], dtype=torch.float16)
        audio_tensor = torch.tensor(audio, dtype=torch.float16)
        return audio_tensor, numeric_label
    
    def _pad_or_truncate(self, audio):
        """Ensures that all audio samples have the same fixed length"""
        if len(audio) < self.fixed_length:
            pad_size = self.fixed_length - len(audio)
            audio = np.pad(audio, (0, pad_size), mode='constant')
        else:
            audio = audio[:self.fixed_length] 
        return audio





class CustomDataset(Dataset):
     '''
     Class for spectrogram image dataset
     '''

     def __init__(self, csv_file, img_folder, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with 'image_path' and 'label' columns.
            img_folder (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform

     def __len__(self):
        return len(self.data_frame)

     def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, str(self.data_frame.iloc[idx, 0]) + '.jpg')
        label = self.data_frame['Y'][idx] 
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
