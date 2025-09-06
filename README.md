CNN Image Classification (TensorFlow)
Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model learns to identify 10 different classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

Features

Loads and preprocesses the CIFAR-10 dataset

Builds a CNN model with multiple convolutional and dense layers

Trains the model and validates its performance

Evaluates test accuracy

Plots accuracy and loss graphs

Makes predictions on sample test images

Installation

Make sure you have Python 3.13+ installed.

Clone the repository:

git clone <your-repo-link>


Install required libraries:

pip install tensorflow matplotlib numpy

Usage

Run the script to train and evaluate the CNN model:

python CNN_image_classification.py


The script will:

Download the CIFAR-10 dataset automatically

Train the CNN for 10 epochs

Display training/validation accuracy and loss graphs

Print test accuracy and predictions for sample images

Dependencies

TensorFlow

NumPy

Matplotlib

Screenshots

(Optional: Add screenshots of accuracy/loss graphs or predictions here)

Results

Expected test accuracy: ≈70%
The model correctly classifies images across the 10 CIFAR-10 classes.
