# 🧠 Multi-Task NLP using T5 Transformer

This project implements a custom NLP model using a masked T5 transformer for **Sentiment Analysis**, **Emotion Detection**, and **Text Summarization** from website comments. It uses three separate pipelines for each task, trained with custom loops and deployed via a Flask app. This is only the model for app and backed i will provide the github link below

## 🚀 Features

* Transformer-based (T5) multi-task NLP system
* Custom training loops for classification and summarization
* Three independently trained pipelines with tailored preprocessing
* GPU training using Kaggle
* Flask-based local deployment

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:harshitgupta04022004/Sentiment_detection_model.git
cd nlp-t5-multitask
```

### 2. Create a virtual environment

```bash
conda create --name nlp-t5-env python=3.9
conda activate nlp-t5-env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

---

## 🧪 Model Training Details

* Model: Masked T5 Transformer
* Task-Specific Pipelines:

  * Sentiment & Emotion Detection: Custom classification loss
  * Summarization: Direct decoding from T5
* Training: Kaggle Notebooks with GPU
* Deployment: Flask (Localhost), model weights loaded dynamically

---

## 📁 Project Structure

```
├── data/                  # Data for training and testing the models
├── model_notebooks/       # Notebooks for model development and checkpoint testing
├── results/               # Saved model results, logs, and outputs
├── app.py                 # Flask app for local deployment
├── checkpoint             # Result of checkpoint 2
├── comments.json          # Sample input data for testing
├── load model.ipynb       # Load and predict model outputs
├── req.txt                # (Ignore) Older requirement file
├── requirement.txt        # Correct file for installing dependencies
├── README.md              # Project README


---

## 🧑‍💻 Author

**Harshit Gupta**
Contributions welcome! Feel free to fork or open issues.

---
