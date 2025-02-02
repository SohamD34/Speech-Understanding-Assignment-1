import os
import librosa


def import_audio_files(folder_path):

    X = []
    
    for filename in os.listdir(folder_path): 
        if filename.endswith('.mp3'):  
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=4000)
            X.append((filename, audio, sr))

    return X
