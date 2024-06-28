import pandas as pd
# TF for image segmentation model
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
from netcal.metrics import ECE, MCE, ACE
from netcal.presentation import ReliabilityDiagram
from netcal.scaling import TemperatureScaling
from sklearn.metrics import classification_report, accuracy_score
from netcal.binning import IsotonicRegression

import librosa


# Kaggle for this YAMNET : https://www.kaggle.com/code/gauravduttakiit/yamnet-environmental-sound-classification-50/notebook
model = tensorflow.saved_model.load( './' )
classes = [  "airplane" ,  "breathing" ,  "brushing_teeth" ,  "can_opening" ,  "car_horn" ,  "cat" ,  "chainsaw" ,  "chirping_birds" ,  "church_bells" ,  "clapping" ,  "clock_alarm" ,  "clock_tick" ,  "coughing" ,  "cow" ,  "crackling_fire" ,  "crickets" ,  "crow" ,  "crying_baby" ,  "dog" ,  "door_wood_creaks" ,  "door_wood_knock" ,  "drinking_sipping" ,  "engine" ,  "fireworks" ,  "footsteps" ,  "frog" ,  "glass_breaking" ,  "hand_saw" ,  "helicopter" ,  "hen" ,  "insects" ,  "keyboard_typing" ,  "laughing" ,  "mouse_click" ,  "pig" ,  "pouring_water" ,  "rain" ,  "rooster" ,  "sea_waves" ,  "sheep" ,  "siren" ,  "sneezing" ,  "snoring" ,  "thunderstorm" ,  "toilet_flush" ,  "train" ,  "vacuum_cleaner" ,  "washing_machine" ,  "water_drops" ,  "wind" ,  ]
Id=[]
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./esc50/test'):
    for filename in filenames:
        Id.append(os.path.join(dirname, filename))
test=pd.DataFrame()
test=test.assign(filename=Id)
test['label']=test['filename']
test['label']=test['label'].str.replace('./esc50/test\\','')
test['label'] = test['label'].str.split('\\').str[0]
test.head()
model_groundtruth_arg = np.array([])
# Loop over test DataFrame
for label in test['label']:
    index = classes.index(label)
    if len(model_groundtruth_arg) == 0:
        model_groundtruth_arg = np.array([index])
    else:
        model_groundtruth_arg = np.append(model_groundtruth_arg, index)


predictions_test = np.array([])
result=[]
for i in test.filename:
    waveform , sr = librosa.load(i , sr=16000)
    if waveform.shape[0] % 16000 != 0:
        waveform = np.concatenate([waveform, np.zeros(16000)])
    inp = tensorflow.constant( np.array([waveform]) , dtype='float32'  )
    class_scores = model( inp )[0].numpy()
    if len(predictions_test) == 0:  # If predictions_test is empty, directly assign class_scores
        predictions_test = class_scores
    else:  # Otherwise, stack class_scores vertically
        predictions_test = np.vstack((predictions_test, class_scores))
    result.append( classes[  class_scores.argmax()])

#******************************************************************
model_prediction_arg = np.argmax(predictions_test, axis=1)
model_accuracy = accuracy_score(model_prediction_arg, model_groundtruth_arg)
print(f'Model Accuracy: {model_accuracy}')

# Calculate confidence of model predictions
model_probabilities = predictions_test
model_confidence = np.max(model_probabilities, axis=1)
average_model_confidence = np.mean(model_confidence)
print(f'Average Model Confidence per Prediction: {average_model_confidence}')

# Calculate ECE
n_bins=10

ece = ECE(n_bins)
ece_uncalibrated_score = ece.measure(model_probabilities, model_groundtruth_arg)
mce = MCE(n_bins)
mce_uncalibrated_score = mce.measure(model_probabilities, model_groundtruth_arg)
ace = ACE(n_bins)
ace_uncalibrated_score = ace.measure(model_probabilities, model_groundtruth_arg)
print(f'Uncalibrated Accuracy: {model_accuracy}, ECE: {ece_uncalibrated_score}, MCE: {mce_uncalibrated_score}, ACE: {ace_uncalibrated_score}')

# Plot all Temp scaling and Uncalibrated
diagram = ReliabilityDiagram(n_bins)

diagram = ReliabilityDiagram(n_bins)
diagram.plot(model_probabilities, model_groundtruth_arg, title_suffix='Uncalibrated Model')  # visualize miscalibration of uncalibrated
plt.show()


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
temperature.weights[0] = 1.75
calibrated_1_25 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_25 = ece.measure(calibrated_1_25, model_groundtruth_arg)
mce_calibrated_score_1_25 = mce.measure(calibrated_1_25, model_groundtruth_arg)
ace_calibrated_score_1_25 = ace.measure(calibrated_1_25, model_groundtruth_arg)
print(f'Temp Scaling 1.25 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_25}, MCE: {mce_calibrated_score_1_25}, ACE: {ace_calibrated_score_1_25}')

# T=1.13
temperature.weights[0] = 1.78
calibrated_1_1 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_1 = ece.measure(calibrated_1_1, model_groundtruth_arg)
mce_calibrated_score_1_1 = mce.measure(calibrated_1_1, model_groundtruth_arg)
ace_calibrated_score_1_1 = ace.measure(calibrated_1_1, model_groundtruth_arg)
print(f'Temp Scaling 1.1 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_1}, MCE: {mce_calibrated_score_1_1}, ACE: {ace_calibrated_score_1_1}')

# T=1.15
temperature.weights[0] = 1.82
calibrated_1_19 = temperature.transform(model_probabilities)
# Calculate error metrices
ece_calibrated_score_1_19 = ece.measure(calibrated_1_19, model_groundtruth_arg)
mce_calibrated_score_1_19 = mce.measure(calibrated_1_19, model_groundtruth_arg)
ace_calibrated_score_1_19 = ace.measure(calibrated_1_19, model_groundtruth_arg)
print(f'Temp Scaling 1.19 Accuracy: {model_accuracy}, ECE: {ece_calibrated_score_1_19}, MCE: {mce_calibrated_score_1_19}, ACE: {ace_calibrated_score_1_19}')


# Isotonic Regression
iso = IsotonicRegression()
iso.fit(model_probabilities, model_groundtruth_arg)
iso_probabilities = iso.transform(model_probabilities)
# Calculate error metrices
ece_iso_calibrated_score = ece.measure(iso_probabilities, model_groundtruth_arg)
mce_iso_calibrated_score = mce.measure(iso_probabilities, model_groundtruth_arg)
ace_iso_calibrated_score = ace.measure(iso_probabilities, model_groundtruth_arg)
# Plot
iso_prediction_arg = np.argmax(iso_probabilities, axis=1)
iso_acc = accuracy_score(iso_prediction_arg, model_groundtruth_arg)
print(f'Isotonic Regression Accuracy: {iso_acc}, ECE: {ece_iso_calibrated_score}, MCE: {mce_iso_calibrated_score}, ACE: {ace_iso_calibrated_score}')

# Plot all Temp scaling and Uncalibrated
diagram = ReliabilityDiagram(n_bins)

diagram = ReliabilityDiagram(n_bins)
diagram.plot(model_probabilities, model_groundtruth_arg, title_suffix='Uncalibrated Model')  # visualize miscalibration of uncalibrated
diagram.plot(calibrated_1_25, model_groundtruth_arg, title_suffix=f'Temperature Scaling - T={0.875}')   # visualize miscalibration of calibrated
#diagram.plot(calibrated_1_1, model_groundtruth_arg, title_suffix=f'Temperature Scaling - T={0.98}')   # visualize miscalibration of calibrated
#diagram.plot(calibrated_1_19, model_groundtruth_arg, title_suffix=f'Temperature Scaling - T={0.97}')   # visualize miscalibration of calibrated
diagram.plot(iso_probabilities, model_groundtruth_arg, title_suffix=f'Isotonic Regression')   # visualize miscalibration of calibrated

plt.show()

