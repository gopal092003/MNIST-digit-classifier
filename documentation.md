# 📘 MNIST Digit Classifier — Documentation

---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify handwritten digits from the MNIST dataset.

The system is designed with a **modular architecture**, separating concerns such as data loading, model definition, training, and evaluation for scalability and maintainability.

---

## 🧠 Model Architecture

The model is a **2-layer CNN followed by fully connected layers**.

### 🔹 Input

* Shape: `(1 × 28 × 28)`
* Grayscale image

---

### 🔹 Convolutional Block 1

* `Conv2D`: 1 → 32 channels, kernel size = 3×3, padding = 1
* Activation: ReLU
* `MaxPool2D`: 2×2
* Output: `(32 × 14 × 14)`

---

### 🔹 Convolutional Block 2

* `Conv2D`: 32 → 64 channels, kernel size = 3×3, padding = 1
* Activation: ReLU
* `MaxPool2D`: 2×2
* Output: `(64 × 7 × 7)`

---

### 🔹 Fully Connected Layers

* Flatten: `64 × 7 × 7 = 3136`
* `Linear`: 3136 → 128
* Activation: ReLU
* Dropout: `p = 0.25`
* `Linear`: 128 → 10 (output classes)

---

## ⚙️ Configuration

All hyperparameters are defined in:

```bash
configs/config.yaml
```

### Example:

```yaml
batch_size: 64
test_batch_size: 1000
learning_rate: 0.001
epochs: 5
```

---

## 📂 Code Structure

### 🔹 `src/data/loader.py`

Handles:

* Dataset downloading (MNIST)
* Transformations (normalization)
* DataLoader creation

---

### 🔹 `src/models/cnn.py`

Defines:

* CNN architecture
* Forward propagation logic

---

### 🔹 `src/training/train.py`

Responsible for:

* Model initialization
* Training loop
* Loss computation
* Backpropagation
* Optimizer updates
* Model saving

---

### 🔹 `src/evaluation/evaluate.py`

Handles:

* Model loading
* Evaluation on test dataset
* Accuracy computation

---

### 🔹 `src/utils/visualize.py`

Provides:

* Visualization of predictions using matplotlib

---

## 🏋️ Training Pipeline

1. Load configuration
2. Load dataset using DataLoader
3. Initialize model, loss, optimizer
4. Iterate over epochs:

   * Forward pass
   * Compute loss
   * Backward pass
   * Update weights
5. Save trained model

---

## 📈 Evaluation Pipeline

1. Load trained model weights
2. Switch model to evaluation mode
3. Disable gradient computation
4. Perform inference on test set
5. Compute accuracy

---

## 📊 Results

* **Test Accuracy:** ~99.26%
* **Training Loss:** ~0.026 after 5 epochs
* Model converges quickly and generalizes well

---

## 🔧 Device Support

* Automatically uses:

  * GPU (CUDA) if available
  * CPU otherwise

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📦 Dependencies

* torch
* torchvision
* matplotlib
* pyyaml

Install via:

```bash
pip install -r requirements.txt
```

---

## 🔍 Key Design Decisions

### ✅ Normalization

```python
Normalize((0.1307,), (0.3081,))
```

Improves convergence by standardizing input distribution.

---

### ✅ Dropout (0.25)

* Prevents overfitting
* Improves generalization

---

### ✅ Modular Design

* Easy to extend
* Clean separation of concerns
* Industry-style structure

---

## 🚀 How to Run

### Training:

```bash
python -m src.training.train
```

### Evaluation:

```bash
python -m src.evaluation.evaluate
```

---

## 🔮 Future Enhancements

* Confusion Matrix & classification report
* Grad-CAM visualization
* Hyperparameter tuning
* TensorBoard integration
* Model deployment (Streamlit / FastAPI)

---

## 📌 Conclusion

This project demonstrates a **high-performance CNN pipeline** for image classification using PyTorch, achieving strong accuracy with a clean and scalable codebase.

It serves as a solid foundation for:

* Computer vision projects
* Deep learning experimentation
* Production-ready ML pipelines

---
