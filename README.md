# 🎧 Signal Classification with Audio Features and Deep Learning Technics

This project focuses on the classification of audio commands using feature extraction and machine learning.  
It applies signal processing techniques to audio data, and evaluates both traditional ML classifiers and deep learning architectures such as CNN and LSTM.

---

## 📂 Dataset

- Custom dataset of `.wav` audio files organized by class folders
- Sampling rate fixed at **16,000 Hz**
- One class ("inohom") is excluded from training

---

## 🔍 Feature Extraction

Extracted audio features include:
- **MFCCs** (Mel-Frequency Cepstral Coefficients)
- **Chroma** features
- **Spectral contrast, centroid, rolloff**
- **Zero-crossing rate**
- **RMS energy**
- **Delta and delta-delta of MFCCs**
- **Onset envelope**

Features are aggregated using mean and standard deviation for each file.

---

## ⚙️ Machine Learning Models

After applying SMOTE to balance the dataset and scaling the features, the following models are trained using stratified 10-fold cross-validation:

- Decision Tree  
- Random Forest  
- K-Nearest Neighbors  
- Naive Bayes  
- SVM (RBF kernel)  
- Logistic Regression

The best-performing model is selected and evaluated on the test set.

---

## 🧠 Deep Learning Models

Two deep architectures are implemented:

### 🧮 LSTM (Bidirectional)
- Stacked BiLSTM layers
- Dense + Dropout
- Trained on reshaped fixed-length feature arrays

### 🧠 CNN (2D)
- 2D convolution layers
- BatchNorm + MaxPooling
- Trained directly on MFCC images

---

## 📊 Evaluation Metrics

- Classification accuracy
- Confusion matrix visualization
- `classification_report()` from sklearn
- Inference time measurements

---

## 📌 Notes

- SMOTE is used for class balancing before splitting the data
- LSTM input is padded/cropped to match expected input dimensions
- CNN expects MFCCs of shape (40, 100)

---

## 🧪 Result Highlights

- CNN and LSTM outperform classical ML models
- Confusion matrices show clear improvements with deep models
- Full training and testing pipelines included

---

## 🗂️ Files

- `Signal Processing and Machine Learning Final Project.ipynb`: full Colab implementation
- `cnn_model_voice.h5`, `lstm_model.h5`: saved deep learning models (optional)
- Dataset should be structured in folders per class under a defined path

---

## 💻 Dependencies

```bash
pip install numpy librosa scikit-learn imbalanced-learn xgboost lightgbm seaborn tensorflow
