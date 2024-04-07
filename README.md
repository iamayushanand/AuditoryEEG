# AuditoryEEG
Source code for Speech Understanding course minor project based on Auditory EEG match mismatch prediction problem.

## Modules

### 1. [dataset.py](src/dataset.py)

The `dataset.py` module is a part of the project's data processing pipeline. It provides classes and functions for handling datasets containing EEG  recordings along with corresponding event information and audio stimuli.

* **Data Loading:** The Dataset class in this module facilitates the loading of EEG recordings and associated event information from specified directories within the dataset.

* **Data Processing:** It offers methods for preprocessing EEG data, such as extracting features using techniques like wavelet transforms, and extracting features from audio signals using techniques like MFCC (Mel-Frequency Cepstral Coefficients).

* **Dataset Preparation:** The prepare_dataset method prepares a dataset by loading a specified number of samples, extracting features, and organizing them into arrays suitable for machine learning tasks.

> _To use `dataset.py`, you can import it into your project and create an instance of the Dataset class. You can then utilize its methods for loading, preprocessing, preparing, and splitting datasets for EEG-based machine learning tasks._

### 2. [evaluator.py](src/evaluator.py)

The `evaluator.py` module is a project evaluation pipeline component. It provides classes and functions for evaluating machine learning models on datasets processed by the `dataset.py` module.

* **Model Evaluation:** The `Evaluator` class in this module is designed to evaluate the performance of various machine learning models on datasets prepared using the Dataset class from the `dataset.py` module.

* **Model Selection:** It allows for evaluating models such as Random Forest Classifier (`rfc`) and Logistic Regression (`lr`) on the dataset. Additional models can be easily integrated into the evaluation process.

> _To utilize evaluator.py, you need to have a prepared dataset instance from the Dataset class. After importing the Evaluator class into your project, you can create an instance of it, passing the dataset as a parameter. You can then call the get_evaluation method, optionally specifying a list of models to evaluate. This method returns a dictionary containing the evaluation results for each model._

``` python
from .dataset import Dataset
from .evaluator import Evaluator

# Create dataset instance
dataset = Dataset()

# Create evaluator instance
evaluator = Evaluator(dataset)

# Get evaluation results
evaluation_results = evaluator.get_evaluation(model_list=['rfc', 'lr'])
```
### 3. [feature_extractor.py](src/feature_extractor.py)

The `feature_extractor.py` module is a component of the project's data processing pipeline, specifically focusing on feature extraction from EEG recordings and audio signals.

**EEG Feature Extraction:**
  - **Short-Time Fourier Transform (STFT) Features:**
    - The `stft_features` function computes STFT features from EEG data after applying Independent Component Analysis (ICA) for dimensionality reduction.
  - **Wavelet Features:**
    - The `wavelet_features` function computes wavelet-based features from EEG data, including energy, entropy, mean, and standard deviation at different levels.
  
**Audio Feature Extraction:**
  - **Mel-Frequency Cepstral Coefficients (MFCC) Features:**
    - The `mfcc_features` function computes MFCC features from audio signals.

> _To utilize feature_extractor.py, you can import it into your project. Depending on your data, you can choose the appropriate function for feature extraction._
```python
from .feature_extractor import EEG_Features, Audio_Features

# Example usage for EEG feature extraction
eeg_data = ...  # EEG data as numpy array
stft_feats = EEG_Features.stft_features(eeg_data)
wavelet_feats = EEG_Features.wavelet_features(eeg_data)

# Example usage for audio feature extraction
audio_signal = ...  # Audio signal as numpy array
mfcc_feats = Audio_Features.mfcc_features(audio_signal)
```
### 4. [test.py](src/test.py)

The `test.py` script is designed to run evaluations on machine learning models using the `Evaluator` class from `evaluator.py` and datasets from the `Dataset` class in `dataset.py`.

> _To use test.py, you can simply run the script. It will print out the evaluation results in a pandas DataFrame format, specifically evaluating the Random Forest Classifier (rfc) model by default._
```python
python test.py
```












