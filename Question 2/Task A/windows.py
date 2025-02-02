import numpy as np

def hann_window(n: int):
    window = np.sin(np.pi * np.arange(n) / n) ** 2
    return window

def hamming_window(n : int):
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)
    return window

def rectangular_window(n : int):
    window = np.ones(n)
    window = window / np.linalg.norm(window)
    return window


def get_window(window_name:str, length: int):
    if window_name == 'hann':
        return hann_window(length)
        
    elif window_name =="hamming":
        return hamming_window(length)
        
    return rectangular_window(length) 