import pandas as pd
# TF for image segmentation model
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
from netcal.metrics import ECE, MCE, ACE
from netcal.presentation import ReliabilityDiagram
from sklearn.metrics import classification_report, accuracy_score
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