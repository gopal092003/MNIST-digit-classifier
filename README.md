# 🧠 MNIST Digit Classifier (PyTorch CNN)

A deep learning project implementing a **Convolutional Neural Network (CNN)** for handwritten digit classification on the MNIST dataset using PyTorch.

---

## 🚀 Features

* ✅ CNN-based architecture (Conv + Pool + FC)
* ✅ Dropout for regularization
* ✅ Normalized input pipeline
* ✅ Modular project structure (production-ready)
* ✅ Cross-platform execution (Windows + Linux)
* ✅ **~99.26% test accuracy**
* ✅ Visualization of predictions

---

## 🧱 Model Architecture

![Architecture](assets/Architecture.png)

**Architecture Details:**

* Input: `1 × 28 × 28`
* Conv1: `32 filters, 3×3` → ReLU → MaxPool → `32 × 14 × 14`
* Conv2: `64 filters, 3×3` → ReLU → MaxPool → `64 × 7 × 7`
* Flatten → `3136`
* FC1: `128 neurons` → ReLU → Dropout (0.25)
* FC2: `10 output classes`

---

## 📊 Results

![Results](assets/Results.png)

* **Test Accuracy:** `99.26%`
* **Final Training Loss:** `~0.026`
* Stable convergence within **5 epochs**

---

## 📁 Project Structure

```bash
mnist-digit-classifier/
│
├── src/
│   ├── data/        # Data loading & preprocessing
│   ├── models/      # CNN architecture
│   ├── training/    # Training loop
│   ├── evaluation/  # Evaluation logic
│   └── utils/       # Visualization
│
├── configs/         # Hyperparameters
├── scripts/         # Run scripts
├── outputs/         # Saved models
├── notebooks/       # Jupyter experiments
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd mnist-digit-classifier
pip install -r requirements.txt
```

---

## 🏋️ Training

### ✅ Recommended (cross-platform)

```bash
python -m src.training.train
```

### 🐧 Linux / Mac

```bash
bash scripts/train.sh
```

### 🪟 Windows

```bash
python src/training/train.py
```

---

## 📈 Evaluation

```bash
python -m src.evaluation.evaluate
```

Output:

```bash
Test Accuracy: 99.26%
```

---

## 🖼️ Sample Predictions

The model:

* Correctly classifies handwritten digits
* Generalizes well to different writing styles
* Achieves high confidence predictions

---

## 🧠 Tech Stack

* Python
* PyTorch
* Torchvision
* Matplotlib

---

## 🔮 Future Improvements

* 📊 Confusion Matrix & Precision/Recall
* 🔥 Grad-CAM visualization
* 📈 Training curves (loss/accuracy plots)
* 🧪 Hyperparameter tuning
* 🌐 Deploy as web app (Streamlit)

---

## 📌 Notes

* Dataset auto-downloads (no manual setup needed)
* GPU used automatically if available
* Modular design → easy to extend and scale

---

## ⭐ Acknowledgements

* MNIST Dataset
* PyTorch Documentation

---

## 💡 Author

**Gopal Gupta**

---

If you found this useful, consider giving it a ⭐ on GitHub!