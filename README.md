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

In our deep dive into the literature, we mainly focused on post-training methods. Here are some of the leading methods of this type described in the literature:

### Histogram Binning

Histogram binning is a non-parametric method that calibrates the model by partitioning the probability space into bins and adjusting the predicted probabilities within each bin.

**Formula:**
$\[ P(x \in B_i) = \frac{|B_i|}{N} \]$
Where $\( B_i \)$ is the number of samples in bin $\( i \)$, and $\( N \)$ is the total number of samples.

### Isotonic Regression

Isotonic regression is a non-parametric calibration method that fits a piecewise constant non-decreasing function to the predicted probabilities.

**Formula:**
$\[ \hat{p}$ = \min_{$\hat{p}$_1 \leq \hat{p}_2 \leq \ldots \leq \hat{p}_n} \sum_{i=1}^{n} (y_i - \hat{p}_i)^2 \]
Where $\( y_i \)$ are the true labels and $\( \hat{p}_i \)$ are the predicted probabilities.

### Platt Calibration

Platt calibration uses logistic regression to map the predicted probabilities to calibrated probabilities.

**Formula:**
$\[ P(y=1|x) = \frac{1}{1 + \exp(Ax + B)} \]$
Where $\( A \)$ and $\( B \)$ are parameters learned from the validation set.

### Temperature Scaling

Temperature scaling is a simple extension of Platt calibration, where a single parameter $\( T \)$ (temperature) is used to scale the logits before applying the softmax function.

**Formula:**
$\[ \sigma\left(\frac{z}{T}\right) \]$
Where $\( z \)$ are the logits and $\( T \)$ is the temperature parameter.
