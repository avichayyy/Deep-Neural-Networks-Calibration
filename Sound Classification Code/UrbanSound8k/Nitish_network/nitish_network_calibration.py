import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from netcal.scaling import TemperatureScaling
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from netcal.metrics import ACE
from netcal.metrics import MCE
from netcal.binning import HistogramBinning
from netcal.binning import IsotonicRegression

# Kaggle for this:
# https://www.kaggle.com/code/sainitishmitta04/audio-classifier-deep-learning-90-accuracy/output?select=model.png
# Load the model
model_path = r"model.h5"
model = load_model(model_path)

# Function to load the UrbanSound8K dataset
def load_urbansound8k_dataset(dataset_path):
    csv_file = os.path.join(dataset_path, "metadata", "UrbanSound8K.csv")
    data = pd.read_csv(csv_file)
    return data

# Load the UrbanSound8K dataset
dataset_path = r".\urbansound8k"
urbansound_data = load_urbansound8k_dataset(dataset_path)

# Get the list of audio files
audio_files = urbansound_data['slice_file_name'].values.tolist()

# Function to preprocess audio files
def preprocess_audio(audio_file):
    fold_number = urbansound_data.loc[urbansound_data['slice_file_name'] == audio_file, 'fold'].values[0]
    audio, sample_rate = librosa.load(os.path.join(dataset_path, "audio", f"fold{fold_number}", audio_file), sr=None)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    return mfccs_scaled_features

# Label encoder to convert class labels to integers
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

audio_dataset_path=r'.\urbansound8k\audio'
metadata = pd.read_csv(r'.\urbansound8k\metadata\UrbanSound8K.csv')
metadata.head()
### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for i,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature', 'class'])
extracted_features_df.head()

label_encoder = LabelEncoder()

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)
model_prediction_arg = np.argmax(model.predict(X_test), axis=1)
model_groundtruth_arg = np.argmax(y_test, axis=1)
model_accuracy = accuracy_score(model_prediction_arg, model_groundtruth_arg)
print(f'Model Accuracy: {model_accuracy}')

# Calculate confidence of model predictions
model_probabilities = model.predict(X_test)
model_confidence = np.max(model_probabilities, axis=1)
average_model_confidence = np.mean(model_confidence)

print(f'Average Model Confidence per Prediction: {average_model_confidence}')
n_bins = 10

# Temperature Scaling
temperature = TemperatureScaling()
temperature.fit(model_probabilities, model_groundtruth_arg)
# Calculate error metrices
ece = ECE(n_bins)
ece_uncalibrated_score = ece.measure(model_probabilities, model_groundtruth_arg)
mce = MCE(n_bins)
mce_uncalibrated_score = mce.measure(model_probabilities, model_groundtruth_arg)
ace = ACE(n_bins)
ace_uncalibrated_score = ace.measure(model_probabilities, model_groundtruth_arg)
print(f'Uncalibrated Accuracy: {model_accuracy}, ECE: {ece_uncalibrated_score}, MCE: {mce_uncalibrated_score}, ACE: {ace_uncalibrated_score}')


# T=1.1
temperature.weights[0] = 1.1
calibrated_1_1 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_1 = ece.measure(calibrated_1_1, model_groundtruth_arg)
mce_calibrated_score_1_1 = mce.measure(calibrated_1_1, model_groundtruth_arg)
ace_calibrated_score_1_1 = ace.measure(calibrated_1_1, model_groundtruth_arg)
print(f'Temp Scaling 1.1 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_1}, MCE: {mce_calibrated_score_1_1}, ACE: {ace_calibrated_score_1_1}')

# T=1.13
temperature.weights[0] = 1.13
calibrated_1_13 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_13 = ece.measure(calibrated_1_13, model_groundtruth_arg)
mce_calibrated_score_1_13 = mce.measure(calibrated_1_13, model_groundtruth_arg)
ace_calibrated_score_1_13 = ace.measure(calibrated_1_13, model_groundtruth_arg)
print(f'Temp Scaling 1.13 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_13}, MCE: {mce_calibrated_score_1_13}, ACE: {ace_calibrated_score_1_13}')

# T=1.15
temperature.weights[0] = 1.15
calibrated_1_15 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_15 = ece.measure(calibrated_1_15, model_groundtruth_arg)
mce_calibrated_score_1_15 = mce.measure(calibrated_1_15, model_groundtruth_arg)
ace_calibrated_score_1_15 = ace.measure(calibrated_1_15, model_groundtruth_arg)
print(f'Temp Scaling 1.15 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_15}, MCE: {mce_calibrated_score_1_15}, ACE: {ace_calibrated_score_1_15}')


# Plot all Temp scaling and Uncalibrated
diagram = ReliabilityDiagram(n_bins)

diagram = ReliabilityDiagram(n_bins)
diagram.plot(model_probabilities, model_groundtruth_arg)  # visualize miscalibration of uncalibrated
diagram.plot(calibrated_1_1, model_groundtruth_arg)   # visualize miscalibration of calibrated
diagram.plot(calibrated_1_13, model_groundtruth_arg)   # visualize miscalibration of calibrated
diagram.plot(calibrated_1_15, model_groundtruth_arg)   # visualize miscalibration of calibrated

plt.show()

# Histogram Binning
hist = HistogramBinning()
hist.fit(model_probabilities, model_groundtruth_arg)
hist_probabilities = hist.transform(model_probabilities)
# Calculate error metrices
ece_hist_calibrated_score = ece.measure(hist_probabilities, model_groundtruth_arg)
mce_hist_calibrated_score = mce.measure(hist_probabilities, model_groundtruth_arg)
ace_hist_calibrated_score = ace.measure(hist_probabilities, model_groundtruth_arg)
diagram.plot(hist_probabilities, model_groundtruth_arg)   # visualize miscalibration of calibrated
hist_prediction_arg = np.argmax(hist_probabilities, axis=1)
hist_acc = accuracy_score(hist_prediction_arg, model_groundtruth_arg)
print(f'Histogram Binning Accuracy: {hist_acc}, ECE: {ece_hist_calibrated_score}, MCE: {mce_hist_calibrated_score}, ACE: {ace_hist_calibrated_score}')
plt.show()

# Isotonic Regression
iso = IsotonicRegression()
iso.fit(model_probabilities, model_groundtruth_arg)
iso_probabilities = iso.transform(model_probabilities)
# Calculate error metrices
ece_iso_calibrated_score = ece.measure(iso_probabilities, model_groundtruth_arg)
mce_iso_calibrated_score = mce.measure(iso_probabilities, model_groundtruth_arg)
ace_iso_calibrated_score = ace.measure(iso_probabilities, model_groundtruth_arg)
# Plot
diagram.plot(iso_probabilities, model_groundtruth_arg)   # visualize miscalibration of calibrated
iso_prediction_arg = np.argmax(iso_probabilities, axis=1)
iso_acc = accuracy_score(iso_prediction_arg, model_groundtruth_arg)
print(f'Isotonic Regression Accuracy: {iso_acc}, ECE: {ece_iso_calibrated_score}, MCE: {mce_iso_calibrated_score}, ACE: {ace_iso_calibrated_score}')
plt.show()


