# Speech-Recognition
NLP Final Project /Williams College
# Speech Emotion Recognition

This project aims to classify emotions from speech audio using deep learning techniques. It utilizes a Convolutional Neural Network (CNN) model trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Features](#features)
* [Model](#model)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Introduction

This project focuses on building and training a CNN model to recognize emotions expressed in speech audio. The RAVDESS dataset, which contains recordings of actors expressing different emotions, is used for training and evaluation.

## Dataset

The RAVDESS dataset comprises 7356 files of 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Each statement is produced in 8 different emotional states (neutral, calm, happy, sad, angry, fearful, disgust, and surprised). 

You can find the dataset on Kaggle: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## Features

The following audio features are extracted from the speech audio to train the model:

* **Zero Crossing Rate (ZCR)**
* **Chroma_stft**
* **Mel-Frequency Cepstral Coefficients (MFCCs)**
* **Root Mean Square Energy (RMS)**
* **Mel Spectrogram**

## Model

A Convolutional Neural Network (CNN) is employed for emotion classification. The model consists of convolutional layers, max pooling layers, dropout layers, and dense layers. It is trained using the categorical cross-entropy loss function and the Adam optimizer.

## Usage

1. **Install Dependencies:** Make sure you have the necessary dependencies installed (see [Dependencies](#dependencies)).
2. **Download Dataset:** Download the RAVDESS dataset from Kaggle and extract it to a directory named 'dataset' within the project folder.
3. **Run the Code:** Execute the Python script (e.g., `speech_emotion_recognition.py`) to train the model, evaluate its performance, and visualize the extracted features.

## Dependencies

* Python 3.6+
* TensorFlow
* Keras
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Seaborn
* Matplotlib

You can install the dependencies using pip:

## Results

The model achieves an overall accuracy of **59%** on the test set. Here's a detailed classification report:

          precision    recall  f1-score   support

   angry       0.89      0.63      0.74        79
    calm       0.53      0.84      0.65        77
 disgust       0.65      0.55      0.59        73
    fear       0.63      0.60      0.62        73
   happy       0.56      0.45      0.50        78
 neutral       0.00      0.00      0.00        44
     sad       0.44      0.62      0.51        73
surprise       0.60      0.75      0.67        79

accuracy                           0.59       576


macro avg 0.54 0.55 0.53 576 weighted avg 0.57 0.59 0.57 576

 
**Observations:**

* The model performs well in recognizing 'calm' and 'surprise' emotions, with relatively high recall and f1-scores.
* It struggles with 'neutral' emotion, achieving 0% precision, recall, and f1-score. This indicates the model might need further improvement to accurately identify neutral speech.
* Other emotions show moderate performance, with varying levels of precision, recall, and f1-scores.

## Contributing

Contributions to this project are welcome. You can contribute by:

* Improving the model architecture or hyperparameters.
* Implementing different feature extraction techniques.
