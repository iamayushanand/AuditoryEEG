import librosa
import numpy as np
from sklearn.decomposition import FastICA
from math import floor

class EEG_Features():
    def stft_features(X):
        N_feat = 200    
        n_c = 40
        X_ica = FastICA(n_components=n_c).fit_transform(X.T).T
        stft = np.abs(librosa.stft(X_ica, n_fft=128))
        #print(stft.shape)
        hop_length = 128//4
        X_feats = np.zeros((N_feat, n_c*65))
        for start in range(0, N_feat):
            end = start+3
            sr = 1024
            X_feats[start] = np.mean(stft[:, :, floor(start*sr/hop_length):floor(end*sr/hop_length)], axis=2).flatten()
        return X_feats


class Audio_Features():
    
    def mfcc_features(audio_signal):
        N_feat = 200
        feats = np.zeros((N_feat, 40))
        for start in range(0, N_feat):
            end = start+3
            sr = 48000
            feats[start] = np.mean(librosa.feature.mfcc(y=audio_signal[start*sr:end*sr], n_mfcc=40, sr=sr), axis=-1)
        return feats
