import librosa
import numpy as np
from sklearn.decomposition import FastICA
from math import floor
import pywt
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

    def get_wavelet_avg(data,type_wav='haar', level=1):
        coeffs = pywt.wavedec(data, type_wav,level=level)
        cD_Energy = np.mean([np.sum(np.square(coeffs[i+1]), axis=1) for i in range(level)], axis=0)
        cA_Energy = np.sum(np.square(coeffs[0]), axis=1)
        D_entropy = np.mean([sp.stats.entropy(coeffs[i+1], axis=1)for i in range(level)], axis=0)
        A_entropy = sp.stats.entropy(coeffs[0], axis=1)
        D_mean = np.mean([np.mean(coeffs[i+1], axis=1) for i in range(level)], axis=0)
        A_mean = np.mean(coeffs[0], axis=1)
        D_std = np.mean([np.std(coeffs[i+1], axis=1) for i in range(level)], axis=0)
        A_std = np.std(coeffs[0], axis=1)
        features = np.array([cD_Energy, cA_Energy, D_entropy, A_entropy, D_mean, A_mean, D_std, A_std])
        return features

    def wavelet_features(X):
        N_feat = 100    
        n_c = 64
        hop_length = 128//4
        X_feats = np.zeros((N_feat, n_c*8))
        for start in range(0, N_feat):
            end = start+3
            sr = 1024
            X_feats[start] = EEG_Features.get_wavelet_avg(X[:, floor(start*sr):floor(end*sr)]).flatten()
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
