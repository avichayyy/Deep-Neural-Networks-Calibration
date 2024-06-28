# Deep-Neural-Networks-Calibration
Repository for Deep Neural Network Calibration - Project A at the ECE Faculty at the Technion

![image](https://github.com/avichayyy/Deep-Neural-Networks-Calibration/assets/129785797/9201769a-802a-4847-8584-aa45a09d2cf6)

# About
Deep Neural Networks (DNNs) are a type of learned functions which consist of multiple layers between the input and output layers. These layers consist of neurons, where each connection transfers data from one neuron in a lower layer to another in a higher layer. <br>
Most applications of DNNs are to classify between different samples of certain data in by learning from a labeled dataset consisting of N pairs of (X<sub>i</sub>, y<sub>i</sub>) where X is the input and y is the expected output. <br>
It has been shown that DNNs suffer from miscalibration i.e. misalignment between predicted probabilities and actual outcomes. <br>
This project investigates the phenomenon of miscalibration in Deep Neural Networks (DNNs) when applied to audio datasets.
We have tested several architectures of DNNs on several audio datasets to see if they suffer from miscalibration and found that the problem exists in audio datasets as well. Furthermore, the same results applied to image classification DNNs can be applied to audio DNNs. <br>

# Background

In the literature, there are three main approaches to address the problem of calibration:
1. **Using methods after the training process:** These methods use a validation set to perform calibration. The assumption here is that the validation set and the training set are taken from the same sample.
2. **Using regularization during training:** These methods use regularization terms and generally modify the objective function to achieve better calibration of the model.
3. **Reducing the model's uncertainty:** Although there is no unequivocal definition that can be quantified for "uncertainty" in networks, experiments have shown that increasing the number of samples can lead to better model performance and also reduce calibration.

In our deep dive into the literature, we mainly focused on post-training methods. <br>
Here are some of the leading methods of this type described in the literature:

### Histogram Binning

Histogram binning is a non-parametric method that calibrates the model by partitioning the probability space into bins and adjusting the predicted probabilities within each bin.

**Formula:**
```math
[ P(x \in B_i) = \frac{|B_i|}{N} ]
```
Where $\( B_i \)$ is the number of samples in bin $\( i \)$, and $\( N \)$ is the total number of samples.

### Isotonic Regression

Isotonic regression is a non-parametric calibration method that fits a piecewise constant non-decreasing function to the predicted probabilities.

**Formula:**
```math
\hat{p} = \min_{\hat{p}_1 \leq \hat{p}_2 \leq \ldots \leq \hat{p}_n} \sum_{i=1}^{n} (y_i - \hat{p}_i)^2
```
Where $\( y_i \)$ are the true labels and $\( \hat{p}_i \)$ are the predicted probabilities.

### Platt Calibration

Platt calibration uses logistic regression to map the predicted probabilities to calibrated probabilities.

**Formula:**
```math
[ P(y=1|x) = \frac{1}{1 + \exp(Ax + B)} ]
```
Where $\( A \)$ and $\( B \)$ are parameters learned from the validation set.

### Temperature Scaling

Temperature scaling is a simple extension of Platt calibration, where a single parameter $\( T \)$ (temperature) is used to scale the logits before applying the softmax function.

**Formula:**
```math
[ \sigma\left(\frac{z}{T}\right) ]
```
Where $\( z \)$ are the logits and $\( T \)$ is the temperature parameter.

# Datasets Used

We utilized four prominent audio datasets for our experiments: UrbanSound8k, ESC50, FSD50K, and Dcase2017. Below is a brief description of each:

**UrbanSound8k**:  
UrbanSound8k is a dataset containing 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes, such as air conditioners, car horns, and children playing. The dataset is balanced in terms of class representation.

**ESC50**:  
ESC50 is a labeled collection of 2000 environmental audio recordings, prearranged into 50 classes (40 examples per class), including natural sounds like animal noises, water sounds, and human noises. The dataset is balanced across its classes.

**FSD50K**:  
FSD50K is an open dataset of human-labeled sound events containing 51,197 recordings (total duration 108.3 hours) distributed in 200 classes drawn from the AudioSet ontology. This dataset is not balanced, and some classes have significantly more samples than others.

**Dcase2017**:  
Dcase2017 is a dataset for the Detection and Classification of Acoustic Scenes and Events challenge. It includes recordings from 15 different acoustic scenes. The dataset consists of recordings that are balanced across classes.

### Dataset Summary

| Dataset Name  | Num of Classes | Data Samples | Balanced (Yes/No) |
|---------------|----------------|--------------|-------------------|
| UrbanSound8k  | 10             | 8732         | Yes               |
| ESC50         | 50             | 2000         | Yes               |
| FSD50K        | 200            | 51,197       | No                |
| Dcase2017     | 15             | 390          | Yes               |

# Results

### UrbanSound8K

#### Gurbuz Network Results:

![image](https://github.com/avichayyy/Deep-Neural-Networks-Calibration/assets/129785797/e77d112b-0e3a-40b1-b415-efd4964d6d43)


#### Nitish Network Results:

![Nitish Network Results](path/to/your/image2.png)

### ESC-50K

#### Yament Network Results:

![Yament Network Results](path/to/your/image3.png)

#### Resnet Network Results:

![Resnet Network Results](path/to/your/image4.png)

### FSD-50K

#### CNN Network Results:

![CNN Network Results](path/to/your/image5.png)

### Dcase-2017

#### CNN Network Results:

![CNN Network Results](path/to/your/image6.png)


