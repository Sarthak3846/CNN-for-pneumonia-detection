# CNN-for-pneumonia-detection
This project involves building a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The dataset used includes images of chest X-rays classified into two categories: NORMAL and PNEUMONIA.

# Dataset

The dataset contains three folders:

train: Images for training the model
test: Images for testing the model
val: Images for validation

Each folder contains two subfolders:

NORMAL: Chest X-rays of healthy individuals
PNEUMONIA: Chest X-rays of patients with pneumonia

# Model Architecture

The CNN model consists of the following layers:
Convolutional Layers:
Conv2D with 32 filters and ReLU activation
Conv2D with 64 filters and ReLU activation
Conv2D with 128 filters and ReLU activation
Pooling Layers:
MaxPooling2D layers after each Conv2D layer
Flatten Layer:
Converts 2D feature maps into a 1D feature vector
Fully Connected Layers:
Dense layer with 128 neurons and ReLU activation
Dropout layer for regularization
Dense output layer with 1 neuron and sigmoid activation for binary classification

# Training and Validation

The dataset is split into training and validation sets using an 80-20 ratio. The training set is used to train the model, and the validation set monitors the model's performance.

# Loss and Optimizer

Loss Function: Binary Crossentropy
Optimizer: Adam with a learning rate of 0.001

# Training the Model

The model is trained for 10 epochs with a batch size of 32. The images are resized to 150x150 pixels and normalized to a range of [0, 1].
