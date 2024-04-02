
import pandas as pd
import pybdf
from .feature_extractor import EEG_Features, Audio_Features
from tqdm import tqdm
import os
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


class Dataset():

    
    def __init__(self, audio_feat_type="mfcc", eeg_feat_type="spectogram", DATASET_DIR="../../dataset/", STIMULI_DIR="../../dataset/stimuli/"):
        self.DATASET_DIR = DATASET_DIR
        self.STIMULI_DIR = STIMULI_DIR
        self.dataset_df = pd.DataFrame(self.initialize_dataframe())
        self.audio_feature_db = np.array([])
        self.eeg_feature_db = np.array([])
        self.filter_key = ""

    def initialize_dataframe(self):
        dataset_dict = {"events":[],
                        "EEG": []}
        for root, drc, files in os.walk(self.DATASET_DIR):
            for file in files:
                if ("bdf" in file) and ("resting" not in file):
                    dataset_dict["EEG"].append(os.path.join(root, file))
                    event_file_name = '_'.join(file.split("_")[:-1])+"_events.tsv"
                    dataset_dict["events"].append(os.path.join(root, event_file_name))
        return dataset_dict
    
    def load_sample(self, idx):
        current_eeg_file = self.dataset_df["EEG"][idx]
    
        rec = pybdf.bdfRecording(current_eeg_file)
        data = rec.getData()
        sr = rec.sampRate
        

        dataMatrix1 = data['data']
        
        current_event_file = self.dataset_df["events"][idx]
        events = pd.read_csv(current_event_file, sep='\t')
        return events.to_dict(), dataMatrix1[0:64, :]

    def visualise_sample(self, idx):
        _, X = self.load_sample(idx)
        X_ica = FastICA(n_components=64).fit_transform(X.T).T
        time_array1 = np.arange(X.shape[1]) / 1024
        
        fig1 = plt.figure(figsize=(20, 20), dpi=100)

        for i in range(64): 
            ax = fig1.add_subplot(8,8,i+1)
            ax.plot(time_array1, X_ica[i,:])
            ax.set_ylabel('Amplitude ($\mu V$)')
            ax.set_xlabel('Time (s)')
        
    def set_filter_fn(self, key):
        self.filter_key = key
        
    def filter_fn(self, idx):
        current_event_file = self.dataset_df["events"][idx]
        events = pd.read_csv(current_event_file, sep='\t')
        if self.filter_key in events["stim_file"][0]:
            return True
        return False
        
    def audio_signal_from_event(self, events):
        onset = round(events['onset'][0])
        stim_file = np.load(f"{self.STIMULI_DIR}/{events['stim_file'][0][:-3]}")
        audio_signal = stim_file["audio"]
        sr = stim_file["fs"]
        return audio_signal[onset*sr:]

    def prepare_dataset(self, N=5):
        if self.eeg_feature_db.size:
            print("data already prepared!")
            return self.audio_feature_db, self.eeg_feature_db
        
        count = 0
        idx = 0
        pbar = tqdm(total=N)
        while count<N:
            if not self.filter_fn(idx):
                idx += 1
                continue
            events, X = self.load_sample(idx)
            audio_signal = self.audio_signal_from_event(events)
            audio_features = Audio_Features.mfcc_features(audio_signal)
            #eeg_features = EEG_Features.stft_features(X)
            eeg_features = EEG_Features.wavelet_features(X)
            self.audio_feature_db = np.vstack([self.audio_feature_db, audio_features]) if self.audio_feature_db.size else audio_features
            self.eeg_feature_db = np.vstack([self.eeg_feature_db, eeg_features]) if self.eeg_feature_db.size else eeg_features
            count+=1
            pbar.update(count)
            idx += 1
        print("audio size",len(self.audio_feature_db))
        print("eeg size",len(self.eeg_feature_db))
        return self.audio_feature_db, self.eeg_feature_db

    def get_mismatched_feats(self):
        mismatched_audio_feats = self.audio_feature_db.copy()
        mismatched_audio_feats = np.roll(mismatched_audio_feats, 150, axis=0)
        return mismatched_audio_feats

    def get_dataset(self, N=5, train_split=0.8):
        if not self.audio_feature_db.size:
            self.prepare_dataset(N=N)
        
        mismatched_audio_feats = self.get_mismatched_feats()
        train_size = int(len(self.eeg_feature_db)*train_split)
        test_size = int(len(self.eeg_feature_db)*(1-train_split))
        print(len(self.eeg_feature_db))
        X_train = []
        X_test = []
        y_train = np.random.randint(2, size = train_size)
        y_test = np.random.randint(2, size = test_size)
        for i in range(train_size+test_size):
            if i<train_size:
                if y_train[i]==1:
                    X_train.append(np.concatenate((self.eeg_feature_db[i], self.audio_feature_db[i])))
                else:
                    X_train.append(np.concatenate((self.eeg_feature_db[i], mismatched_audio_feats[i])))
            else:
                if y_test[i-train_size]==1:
                    X_test.append(np.concatenate((self.eeg_feature_db[i], self.audio_feature_db[i])))
                else:
                    X_test.append(np.concatenate((self.eeg_feature_db[i], mismatched_audio_feats[i])))

        return np.array(X_train), y_train, np.array(X_test), y_test
