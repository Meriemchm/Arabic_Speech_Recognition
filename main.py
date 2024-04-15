import numpy as np
import scipy.io.wavfile as read
import os
from IPython.display import Audio, display

def label_spoken_word(file_name):

    parts = file_name.split('-')
    

    spoken_word = int(parts[3])
    

    labels = {
        0: 'اعجبني',
        1: 'لم يعجبني',
        2: 'هذا',
        3: 'الفيلم',
        4: 'رائع',
        5: 'مقول',
        6: 'سيئ'
    }
    

    return labels.get(spoken_word, 'Unknown')

# Example usage
folder_name = 'wavs'
file_names = os.listdir(folder_name)
labels = []
for file_name in file_names:
  label = label_spoken_word(file_name)
  labels.append(label)

print(f"Le fichier {file_names[2]} est étiqueté comme : {labels[2]}")





