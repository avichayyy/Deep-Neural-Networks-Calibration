import os
import struct
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from netcal.binning import HistogramBinning
from netcal.binning import IsotonicRegression
from netcal.metrics import ACE
from netcal.metrics import ECE
from netcal.metrics import MCE
from netcal.presentation import ReliabilityDiagram
from netcal.scaling import TemperatureScaling
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/code/badl071/urban-sounds-classification-using-cnns/output - Kaggle of this

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

warnings.filterwarnings("ignore")

def load_data(file_name):
    """Returns a pandas dataframe from a csv file."""
    return pd.read_csv(file_name, header=None)

# Load the model
model_path = r"weights.best.basic_cnn.hdf5"
model = load_model(model_path)

X_test = pd.read_csv('test_data.csv')
y_test = pd.read_csv('test_labels.csv')

X_test = X_test.to_numpy()
X_test = X_test.reshape(X_test.shape[0], 40, 174, 1)

X_val = pd.read_csv('x_val.csv')
y_val = pd.read_csv('y_val.csv')
X_val = X_val.to_numpy()
X_val = X_val.reshape(X_val.shape[0], 40, 174, 1)


model_prediction_arg = np.argmax(model.predict(X_test), axis=1)
model_groundtruth_arg = np.argmax(y_test, axis=1)
model_accuracy = accuracy_score(model_prediction_arg, model_groundtruth_arg)
print(f'Model Accuracy: {model_accuracy}')

# Calculate confidence of model predictions
model_probabilities = model.predict(X_test)
model_confidence = np.max(model_probabilities, axis=1)
average_model_confidence = np.mean(model_confidence)

# Calculate model validation predictions
model_groundtruth_arg_val = np.argmax(y_val, axis=1)

model_probabilities_val = model.predict(X_val)
model_confidence_val = np.max(model_probabilities_val, axis=1)
average_model_confidence_val = np.mean(model_confidence_val)

print(f'Average Model Confidence per Prediction: {average_model_confidence}')
n_bins = 10

# Temperature Scaling
temperature = TemperatureScaling()
temperature.fit(model_probabilities_val, model_groundtruth_arg_val)
# Calculate error metrices
ece = ECE(n_bins)
ece_uncalibrated_score = ece.measure(model_probabilities, model_groundtruth_arg)
mce = MCE(n_bins)
mce_uncalibrated_score = mce.measure(model_probabilities, model_groundtruth_arg)
ace = ACE(n_bins)
ace_uncalibrated_score = ace.measure(model_probabilities, model_groundtruth_arg)
print(f'Uncalibrated Accuracy: {model_accuracy}, ECE: {ece_uncalibrated_score}, MCE: {mce_uncalibrated_score}, ACE: {ace_uncalibrated_score}')


# T=1.1
temperature.weights[0] = 1.25
calibrated_1_25 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_25 = ece.measure(calibrated_1_25, model_groundtruth_arg)
mce_calibrated_score_1_25 = mce.measure(calibrated_1_25, model_groundtruth_arg)
ace_calibrated_score_1_25 = ace.measure(calibrated_1_25, model_groundtruth_arg)
print(f'Temp Scaling 1.25 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_25}, MCE: {mce_calibrated_score_1_25}, ACE: {ace_calibrated_score_1_25}')

# T=1.13
temperature.weights[0] = 1.1
calibrated_1_1 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_1 = ece.measure(calibrated_1_1, model_groundtruth_arg)
mce_calibrated_score_1_1 = mce.measure(calibrated_1_1, model_groundtruth_arg)
ace_calibrated_score_1_1 = ace.measure(calibrated_1_1, model_groundtruth_arg)
print(f'Temp Scaling 1.1 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_1}, MCE: {mce_calibrated_score_1_1}, ACE: {ace_calibrated_score_1_1}')

# T=1.15
temperature.weights[0] = 1.19
calibrated_1_19 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_19 = ece.measure(calibrated_1_19, model_groundtruth_arg)
mce_calibrated_score_1_19 = mce.measure(calibrated_1_19, model_groundtruth_arg)
ace_calibrated_score_1_19 = ace.measure(calibrated_1_19, model_groundtruth_arg)
print(f'Temp Scaling 1.19 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_19}, MCE: {mce_calibrated_score_1_19}, ACE: {ace_calibrated_score_1_19}')


# Plot all Temp scaling and Uncalibrated
diagram = ReliabilityDiagram(n_bins)

diagram = ReliabilityDiagram(n_bins)
diagram.plot(model_probabilities, model_groundtruth_arg, title_suffix='Uncalibrated Model')  # visualize miscalibration of uncalibrated
diagram.plot(calibrated_1_25, model_groundtruth_arg)   # visualize miscalibration of calibrated
diagram.plot(calibrated_1_1, model_groundtruth_arg)   # visualize miscalibration of calibrated
diagram.plot(calibrated_1_19, model_groundtruth_arg)   # visualize miscalibration of calibrated

plt.show()

# Histogram Binning
hist = HistogramBinning()
hist.fit(model_probabilities_val, model_groundtruth_arg_val)
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
iso.fit(model_probabilities_val, model_groundtruth_arg_val)
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
