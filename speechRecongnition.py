import numpy as np
import scipy.io.wavfile as read
import os
from IPython.display import Audio, display
from scipy.fftpack import idct
from sklearn.model_selection import train_test_split



class ArabicSpeechRecognition:
    def __init__(self, file_names):
        self.file_names = file_names
        self.labels = []

    def label_spoken_word(self,file_name):
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

    def Mel2Hz(self, mel):
        return 700 * (np.power(10, mel / 2595) - 1)

    def Hz2Mel(self, freq):
        return 2595 * np.log10(1 + freq / 700)

    def Hz2Ind(self, freq, fs, Tfft):
        return (freq * Tfft / fs).astype(int)

    def hamming(self, T):
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(T) / (T - 1))

    def FiltresMel(self, fs, nf=36, Tfft=512, fmin=100, fmax=8000):
        Indices = self.Hz2Ind(self.Mel2Hz(np.linspace(self.Hz2Mel(fmin), self.Hz2Mel(min(fmax, fs / 2)), nf + 2)), fs, Tfft)
        filtres = np.zeros((int(Tfft / 2), nf))
        for i in range(nf):
            filtres[Indices[i]:Indices[i + 2], i] = self.hamming(Indices[i + 2] - Indices[i])
        return filtres

    def spectrogram(self, x, T, p, Tfft):
        S = []
        for i in range(0, len(x) - T, p):
            S.append(x[i:i + T] * self.hamming(T))  
        S = np.fft.fft(S, Tfft) 
        return np.abs(S), np.angle(S) 

    def mfcc(self, data, filtres, nc=13, T=256, p=64, Tfft=512):
        data = (data[1] - np.mean(data[1])) / np.std(data[1])  
        amp, ph = self.spectrogram(data, T, p, Tfft)
        amp_f = np.log10(np.dot(amp[:, :int(Tfft / 2)], filtres) + 1)
        return idct(amp_f, n=nc, norm='ortho')
    

    def mfcc_processing(self,file_name):
        fs, sgn = read.read(os.path.join('wavs', file_name))
        filtres = self.FiltresMel(fs)
        mfcc_features = self.mfcc((fs, sgn), filtres)
        return mfcc_features
    
    def processing(self):
        
        for file_name in self.file_names:
            label = self.label_spoken_word(file_name)
            self.labels.append(label)
        print('hey')
        print(f"Le fichier {self.file_names[2]} est étiqueté comme : {self.labels[2]}")
        fs, sgn = read.read(os.path.join('wavs', self.file_names[2]))
        filtres = self.FiltresMel(fs)
        mfcc_features = self.mfcc((fs, sgn), filtres)
        print("Mot prononcé:", mfcc_features)


        X_train, X_test, y_train, y_test = train_test_split(self.file_names,self.labels, test_size=0.3, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  

        print(f"Nombre de fichiers d'apprentissage : {len(self.file_names)}")
        print(f"Nombre de fichiers d'apprentissage : {len(X_train)}")
        print(f"Nombre de fichiers de validation : {len(X_test)}") #ajuster les hyperparamètres
        print(f"Nombre de fichiers de test : {len(X_valid)}")

        X_train_mfcc = [self.mfcc_processing(file_name) for file_name in X_train]
        X_valid_mfcc = [self.mfcc_processing(file_name) for file_name in X_valid]
        X_test_mfcc = [self.mfcc_processing(file_name) for file_name in X_test]

        print("Dimensions des features pour l'apprentissage :", len(X_train_mfcc))
        print("Dimensions des features pour la validation :", X_valid_mfcc[0].shape) #13 valeurs dans le vecteur mfcc
        print("Dimensions des features pour les tests :", X_test_mfcc[0].shape)

        return   X_train_mfcc ,X_valid_mfcc,X_test_mfcc,y_test,y_train,y_valid



# ce que je dois faire et mettre es mfcc en xtrain jusqua test et pour y train c'et les label car on fait de la superviser apres j'utilise cnn et apres je calcule les score ett voila         
