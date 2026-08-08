# MNIST Digit Classification using Perceptron, ANN and CNN

A deep learning project for handwritten digit classification using the **MNIST dataset**. This project compares three neural-network approaches — **Perceptron, Artificial Neural Network (ANN), and Convolutional Neural Network (CNN)** — and evaluates their performance on the MNIST test dataset.

## 📌 Project Overview

The objective of this project is to classify handwritten digits from **0 to 9** using different neural-network architectures and compare their classification performance.

The project uses:

* **60,000 training images**
* **10,000 test images**
* Image size: **28 × 28 pixels**
* **10 output classes (0–9)**
* TensorFlow/Keras for neural-network models
* Matplotlib and Seaborn for visualization

The dataset is stored in CSV format, where each image contains 784 pixel features representing the 28×28 image.

---

## 🧠 Models Implemented

### 1. Perceptron

A simple neural-network model consisting of a flattened 28×28 image followed by a 10-class softmax output layer.

**Configuration:**

* Flatten layer
* Dense output layer with 10 neurons
* Softmax activation
* SGD optimizer
* Categorical cross-entropy loss
* 5 epochs

**Test Accuracy: 89.60%**

---

### 2. Artificial Neural Network (ANN)

A fully connected neural network with two hidden layers.

**Architecture:**

```text
Input: 28 × 28
      ↓
Flatten
      ↓
Dense(128, ReLU)
      ↓
Dense(64, ReLU)
      ↓
Dense(10, Softmax)
```

**Configuration:**

* Adam optimizer
* Categorical cross-entropy loss
* Batch size: 32
* Epochs: 10

**Test Accuracy: 97.06%**

---

### 3. Convolutional Neural Network (CNN)

The CNN is designed specifically for image classification and uses convolution and pooling layers to extract spatial features from handwritten digits.

**Architecture:**

```text
Input: 28 × 28 × 1
        ↓
Conv2D(32, 3×3, ReLU)
        ↓
MaxPooling2D(2×2)
        ↓
Conv2D(64, 3×3, ReLU)
        ↓
MaxPooling2D(2×2)
        ↓
Flatten
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(10, Softmax)
```

**Configuration:**

* Adam optimizer
* Categorical cross-entropy loss
* Batch size: 32
* Epochs: 5
* Dropout: 0.5

**Test Accuracy: 98.52%**

---

## 📊 Results

| Model      | Test Accuracy |
| ---------- | ------------: |
| Perceptron |    **89.60%** |
| ANN        |    **97.06%** |
| CNN        |    **98.52%** |

### Performance Comparison

The CNN achieved the highest test accuracy among the three models.

```text
Perceptron   ██████████████████        89.60%
ANN          ███████████████████       97.06%
CNN          ████████████████████      98.52%
```

The notebook also includes a validation-accuracy comparison and a final test-accuracy comparison between all three models.

---

## 🔍 Data Preprocessing

The dataset is loaded from CSV files:

```python
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')
```

The `label` column is separated from the pixel features.

The labels are converted into one-hot encoded vectors using `to_categorical`.

The pixel data is then reshaped according to the model:

```text
ANN / Perceptron
28 × 28

CNN
28 × 28 × 1
```

## The original dataset contains **785 columns**: one label column and 784 pixel columns. No missing values were found in the dataset.

## 📈 Visualizations

The project includes several visual analyses:

* Training vs validation accuracy
* Training vs validation loss
* Validation accuracy comparison
* Side-by-side prediction comparison
* CNN confusion matrix
* Final test accuracy comparison

The side-by-side visualization compares predictions made by the Perceptron, ANN, and CNN on sample test images.

A confusion matrix is also generated for the CNN to analyze classification performance across the ten digit classes.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* Google Colab
* Google Drive

---

## 📂 Dataset

The project uses MNIST-style CSV files:

```text
mnist_train.csv
mnist_test.csv
```

Dataset dimensions used in the project:

```text
Training set: 60,000 × 785
Testing set:  10,000 × 785
```

The notebook loads the dataset from Google Drive and mounts Google Drive through Google Colab.

---

## ▶️ How to Run

### 1. Open the notebook

Open the project notebook in **Google Colab**.

### 2. Mount Google Drive

```python
from google.colab import drive

drive.mount('/content/drive')
```

### 3. Place the datasets in your Google Drive

Make sure the following files are available:

```text
mnist_train.csv
mnist_test.csv
```

### 4. Load the dataset

```python
train_df = pd.read_csv('path/to/mnist_train.csv')
test_df = pd.read_csv('path/to/mnist_test.csv')
```

### 5. Run the notebook

Execute the cells sequentially to:

1. Load the dataset
2. Explore the data
3. Preprocess the images
4. Train the Perceptron
5. Train the ANN
6. Train the CNN
7. Evaluate all models
8. Generate visualizations
9. Compare model performance

---

## 🏆 Conclusion

The experiment demonstrates the progression from a simple linear classifier to increasingly capable neural-network architectures.

The results show:

* **Perceptron:** 89.60%
* **ANN:** 97.06%
* **CNN:** 98.52%

The **CNN performed best**, achieving **98.52% test accuracy**. Its convolution and pooling layers allow it to learn spatial features from the 28×28 handwritten digit images more effectively than the simpler architectures used in this experiment.

---

## 👨‍💻 Project

**MNIST Handwritten Digit Classification**

Built using Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib and Seaborn.
