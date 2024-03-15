
import pandas as pd
import pybdf
from feature_extractor import EEG_Features, Audio_Features
from tqdm import tqdm
import os
import numpy as np

class Dataset():

    
    def __init__(self, DATASET_DIR="../../dataset/", STIMULI_DIR="../../dataset/stimuli/"):
        self.DATASET_DIR = DATASET_DIR
        self.STIMULI_DIR = STIMULI_DIR
        self.dataset_df = pd.DataFrame(self.initialize_dataframe())
        self.audio_feature_db = np.array([])
        self.eeg_feature_db = np.array([])

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

    def audio_signal_from_event(self, events):
        onset = round(events['onset'][0])
        stim_file = np.load(f"{self.STIMULI_DIR}/{events['stim_file'][0][:-3]}")
        audio_signal = stim_file["audio"]
        sr = stim_file["fs"]
        return audio_signal[onset*sr:]

    def prepare_dataset(self):
        
        for idx in tqdm(range(5)):
            events, X = self.load_sample(idx)
            audio_signal = self.audio_signal_from_event(events)
            audio_features = Audio_Features.mfcc_features(audio_signal)
            eeg_features = EEG_Features.stft_features(X)
            self.audio_feature_db = np.vstack([self.audio_feature_db, audio_features]) if self.audio_feature_db.size else audio_features
            self.eeg_feature_db = np.vstack([self.eeg_feature_db, eeg_features]) if self.eeg_feature_db.size else eeg_features
        print(self.audio_feature_db)
        print(self.eeg_feature_db)
        return self.audio_feature_db, self.eeg_feature_db

    def get_dataset(self):
        if not self.audio_feature_db.size:
            self.prepare_dataset()
        
        mismatched_audio_feats = self.audio_feature_db.copy()
        np.random.shuffle(mismatched_audio_feats)
        print(mismatched_audio_feats[0])
        X_train = []
        X_test = []
        y_train = np.random.randint(2, size = 300)
        y_test = np.random.randint(2, size = 200)
        for i in range(500):
            if i<300:
                if y_train[i]==1:
                    X_train.append(np.concatenate(self.eeg_feature_db[i], self.audio_feature_db[i]))
                else:
                    X_train.append(np.concatenate(self.eeg_feature_db[i], mismatched_audio_feats[i]))
            else:
                if y_test[i-300]==1:
                    X_test.append(np.concatenate(self.eeg_feature_db[i], self.audio_feature_db[i]))
                else:
                    X_test.append(np.concatenate(self.eeg_feature_db[i], mismatched_audio_feats[i]))

        return np.array(X_train), y_train, np.array(X_test), y_test