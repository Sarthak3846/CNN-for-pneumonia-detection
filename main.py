import numpy as np 
import pandas as pd 
import os
import cv2
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

train_dir = 'C:/Users/Sarthak Tyagi/Downloads/archive (8)/chest_xray/train'

train_images = []
train_labels = []

for label in os.listdir(train_dir):
    label_path = os.path.join(train_dir, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path,img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150,150))
                train_images.append(img/255.0)
                train_labels.append(0 if label=='NORMAL' else 1)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32,(3,3),activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)
