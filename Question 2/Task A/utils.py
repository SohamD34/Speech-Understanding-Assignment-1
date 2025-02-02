import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pynvml import *
import psutil
import shutil
from scipy.signal import stft

os.environ["TORCH_LOGS"] = ""

import warnings
warnings.filterwarnings('ignore')

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)



def import_audio_files(folder_path, folds, metadata):

    X, Y = [], []
    for fold in range(folds):
        fold_name = f'fold{fold+1}'
        count = 0

        for filename in os.listdir(folder_path + fold_name):
            if count < 600:
                if filename.endswith('.wav'):  
                    count+=1
                    file_path = os.path.join(folder_path + fold_name, filename)
                    audio, sr = librosa.load(file_path, sr=4000)
                    X.append((audio, sr))

                    row = metadata[metadata['file_path'] == file_path]
                    label = row['class']
                    Y.append(label.values[0])
            else:
                break   
        print(f'Fold {fold+1} imported successfully. Total files = {len(X)}')
    return {'X':X,'Y':Y}



def print_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied by GPU:0 = {info.used//1024**2} MB.")
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied by GPU:1 = {info.used//1024**2} MB.")
    
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    for i, cpu in enumerate(cpu_percent):
        print(f"CPU {i}: {cpu}%", end=" ")
    print('\n')



def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Folder '{folder_path}' successfully zipped as '{output_zip}.zip'.")



def plot_all_accuracies(model_list, hann_train_accuracies, hann_val_accuracies, hamming_train_accuracies, hamming_val_accuracies, rect_train_accuracies, rect_val_accuracies):

    x = [2*i+1 for i in range(len(model_list))]
    plt.figure(figsize=(16,5))
    plt.bar([i-0.25 for i in x], hann_train_accuracies, label='Train')
    plt.bar([i+0.25 for i in x], hann_val_accuracies, label='Valid')
    plt.xticks(x,model_list)
    plt.xlabel('Models -->')
    plt.ylabel('Accuracy -->')
    plt.title('Hann Window Accuracy Score')
    plt.grid()
    plt.legend()
    plt.show()

    x = [2*i+1 for i in range(len(model_list))]
    plt.figure(figsize=(16,5))
    plt.bar([i-0.25 for i in x], hamming_train_accuracies, label='Train')
    plt.bar([i+0.25 for i in x], hamming_val_accuracies, label='Valid')
    plt.xticks(x,model_list)
    plt.xlabel('Models -->')
    plt.ylabel('Accuracy -->')
    plt.title('Hamming Window Accuracy Score')
    plt.grid()
    plt.legend()
    plt.show()

    x = [2*i+1 for i in range(len(model_list))]
    plt.figure(figsize=(16,5))
    plt.bar([i-0.25 for i in x], rect_train_accuracies, label='Train')
    plt.bar([i+0.25 for i in x], rect_val_accuracies, label='Valid')
    plt.xticks(x,model_list)
    plt.xlabel('Models -->')
    plt.ylabel('Accuracy -->')
    plt.title('Rectangular Window Accuracy Score')
    plt.grid()
    plt.legend()
    plt.show()