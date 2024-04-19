import numpy as np
import scipy.io.wavfile as read
import os
from IPython.display import Audio, display
from speechRecongnition import ArabicSpeechRecognition
import torch
import torch.nn as nn
import torch.nn.functional as F

# Les MFCC sont spécialement conçus pour capturer les caractéristiques spectrales du signal audio, 
#ce qui les rend très efficaces pour reconnaître les différents phonèmes et mots. 
#Les MFCC sont également robustes aux variations de prononciation et aux bruits de fond, 
#ce qui en fait un choix populaire pour la reconnaissance de la parole.

class MultiTaskModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, stride, fc_input_size, phrase_classes):
        super(MultiTaskModel, self).__init__()
        # Couche convolutive
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_size, stride)
        # Couches entièrement connectées pour chaque tâche
        self.fc_phrase = nn.Linear(fc_input_size, phrase_classes)

    def forward(self, x):
        # Passer à travers la couche convolutive
        x = self.pool(F.relu(self.conv1(x)))
        # Aplatir les données pour les couches entièrement connectées
        x = x.view(x.size(0), -1)
        # Passer à travers les couches entièrement connectées pour chaque tâche
        phrase_output = self.fc_phrase(x)

        return phrase_output
        
        # Évaluation finale sur les données de test
        #model.eval()
        #test_accuracy = evaluate_model(model, X_test)

folder_name = 'wavs'
file_names = os.listdir(folder_name)
speech_recognition = ArabicSpeechRecognition(file_names)
X_train_mfcc ,X_valid_mfcc,X_test_mfcc,y_test,y_train,y_valid = speech_recognition.processing()

# Initialiser le modèle

in_channels = 1  # Nombre de canaux en entrée
out_channels = 16  # Nombre de filtres de la couche convolutive
kernel_size = 3  # Taille du noyau de convolution
pool_size = 2  # Taille de la fenêtre de pooling
stride = 1  # Pas de déplacement du noyau de convolution
fc_input_size = 1000  # Taille de l'entrée des couches entièrement connectées (doit être un entier)
phrase_classes = 7 

model = MultiTaskModel(in_channels, out_channels, kernel_size, pool_size, stride, fc_input_size, 7)
# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
label_to_int = {'اعجبني': 0, 'لم يعجبني': 1, 'هذا': 2, 'الفيلم': 3, 'رائع': 4, 'مقول': 5, 'سيئ': 6}

for _ in range(10):
    model.train()
    for batch_data, batch_labels in zip(X_train_mfcc, y_train):
        optimizer.zero_grad()
        batch_data = torch.tensor(batch_data).float()  # Convertir en flottant

       
        batch_labels = label_to_int.get(batch_labels)
        batch_labels = torch.tensor(batch_labels)
        phrase_output = model(batch_data) #implicite 
        phrase_loss = criterion(phrase_output, batch_labels)
        phrase_loss.backward()
        optimizer.step()










