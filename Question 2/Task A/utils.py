import os
import librosa
import shutil
from pynvml import *
import psutil


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