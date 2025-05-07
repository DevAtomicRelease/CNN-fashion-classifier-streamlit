# CNN-fashion-classifier-streamlit
A CNN-based image classifier using the Fashion-MNIST dataset, deployed with Streamlit for real-time predictions. Includes user feedback for correcting predictions and future model retraining.

Fashion-MNIST Image Classification Project
===========================================

Project Overview:
-----------------
This project implements a CNN model to classify images from the Fashion-MNIST dataset.
It includes data preprocessing, model training (with and without data augmentation),
model evaluation (accuracy, precision, recall, F1-score), and a Streamlit interface for real-time image classification.

Folder Structure:
-----------------
/model/
    - fashion_mnist_cnn_model.h5          ~Trained CNN model file

/notebook/
    - fashion_mnist_training.ipynb        ~Full Colab notebook for training and evaluation

/streamlit_app/
    - app.py                              ~Streamlit app to upload images and get predictions

/corrections/
    - Corrected images (if any user corrections were made)
    - corrections.csv                     ~Log of corrections made by users

/evaluation_outputs/
    - Confusion matrix and training/validation accuracy graphs

/screenshots/
    - Screenshots of Streamlit interface and sample predictions

/project_report/
    - Project_Report.pdf                  ~Complete report detailing methodology, results, and conclusions

Files:
------
- README.txt: This file.
- requirements.txt: List of necessary Python packages.

Setup Instructions:
--------------------
1. Install required libraries:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run streamlit_app/app.py

3. Upload any image (preferably grayscale, 28x28) through the Streamlit interface.

4. If the prediction is wrong, correct it through the interface.
   Corrections are saved automatically for future model retraining.

Optional:
---------
- To retrain the model using corrected images:
  Use the "corrections to CSV" script provided separately to prepare corrected_dataset.csv,
  and merge it with the original training data.

Creator:
-------
@devatomicrealease

