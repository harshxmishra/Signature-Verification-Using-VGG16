# VGG19 Signature Recognition

This project implements a signature recognition system using a VGG19 model. The system is trained on a dataset of signature images and can predict the class of new signature images.

## Project Structure

*   `d/VGG19_Exp_4.ipynb`: Jupyter Notebook containing the complete implementation of the signature recognition system.
*   `signature_one_set/`: Directory containing the training, validation, and test data.
*   `vgg19_trial_4/`: Directory where the trained model is saved.
*   `signature_dataset_flipped/`: Directory containing the training data for inference.
*   `new_test/`: Directory containing the test images for inference.

## Overview

The `d/VGG19_Exp_4.ipynb` notebook performs the following steps:

1.  **Imports Libraries:** Imports necessary libraries such as `tensorflow`, `matplotlib`, `numpy`, and `sklearn`.
2.  **Loads Base Model:** Loads the VGG19 model with pre-trained ImageNet weights, excluding the top classification layer.
3.  **Freezes Layers:** Freezes the first 15 layers of the VGG19 model to prevent them from being trained.
4.  **Adds Custom Layers:** Adds custom layers on top of the VGG19 model, including a flatten layer, dense layers, batch normalization, leaky ReLU activation, and dropout.
5.  **Defines Model:** Defines the complete model with the base VGG19 model and the custom layers.
6.  **Compiles Model:** Compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
7.  **Data Loading and Preprocessing:**
    *   Defines directory paths for training, validation, and test data.
    *   Creates data generators with enhanced augmentation for training data, including rescaling, rotation, shifting, shearing, zooming, vertical flipping, and brightness adjustments.
8.  **Callbacks:**
    *   Defines callbacks for model checkpointing, early stopping, and reducing the learning rate.
9.  **Trains Model:** Trains the model using the training data and evaluates it on the validation data.
10. **Loads Best Model:** Loads the best model weights from the saved checkpoint.
11. **Evaluates Model:** Evaluates the model on the test data and prints the test accuracy.
12. **Generates Predictions:** Generates predictions for the test data and prints a classification report.
13. **Plots Results:** Plots the training and validation accuracy and loss over epochs.
14. **Inference:**
    *   Loads the trained model.
    *   Defines directory paths for training data and new test data.
    *   Creates a class indices dictionary.
    *   Defines a function to preprocess and predict the class of an input image.
    *   Predicts and prints the top 5 predicted classes for each image in the new test directory.

## How to Use

1.  **Set up the environment:** Install the required libraries using `pip install -r requirements.txt` (located in the same directory as the notebook).
2.  **Prepare the data:** Place the training, validation, and test data in the `signature_one_set/` directory, and the new test images in the `new_test/` directory.
3.  **Run the notebook:** Execute the `d/VGG19_Exp_4.ipynb` notebook to train and evaluate the VGG19 model.
4.  **View results:** The training and validation plots will be displayed, and the trained model will be saved in the `vgg19_trial_4/` directory.

## Requirements

*   Python 3.6+
*   TensorFlow
*   Keras
*   Matplotlib
*   Numpy
*   Scikit-learn

## Notes

*   The paths to the data and model checkpoints may need to be adjusted based on your local setup.
*   The training parameters can be adjusted to improve the model's performance.
*   The notebook includes various data augmentation techniques that can be enabled or disabled.
*   The notebook includes functions for plotting training and validation metrics, which can be used to visualize the model's performance.

This project provides a starting point for building a signature recognition system using a VGG19 model.
