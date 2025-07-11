# Diabetic Retinopathy Detection (CNN + Streamlit)

This project applies a Convolutional Neural Network (CNN) for binary image classification to detect Diabetic Retinopathy (DR) from retinal images. The model distinguishes between two classes: DR (presence of diabetic retinopathy) and No_DR (healthy retina). A Streamlit web application is included, featuring three modules: detection, chatbot, and project documentation.

## Model Overview

- Task: Binary classification (DR vs No_DR)
- Model: Custom CNN architecture
- Input size: 224x224 RGB images
- Activation: Sigmoid
- Loss function: Binary crossentropy
- Metrics: Accuracy, Precision, Recall, F1-score, AUC

## Techniques Used

### Preprocessing

- Image resizing and normalisation
- Colour enhancement using OpenCV

### Class Imbalance Handling

- Undersampling of majority class (No_DR)
- Class weights applied during training
- Data augmentation via ImageDataGenerator (rotation, flipping, zooming)

### Training Strategy

- Early stopping
- Balanced dataset selection to improve learning stability

## Final Model Performance

          precision    recall  f1-score   support
      DR       0.95      0.93      0.94       279
   No_DR       0.93      0.95      0.94       271
accuracy                           0.94       550


--- 

# Application 


## Streamlit App Overview

### Detection Module (`detect.py`)

- Loads and uses the trained CNN model
- Randomly selects an image from the test dataset
- Displays model prediction and matches it against ground truth

### Chatbot Module (`chatbot.py`)

- Powered by OpenAI GPT-4
- Ingests DR_ML.md to provide contextualised answers about the project
- Responds to questions on diabetic retinopathy, prevention, prognosis, and treatment

## Project Structure
diabetic_retinopathy_ml/
├── .streamlit/
│ └── secrets.toml
├── pages/
│ ├── app.py
│ ├── detect.py
│ ├── chatbot.py
│ ├── about.py
├── CNN_model.keras
├── DR_ML.md
├── about.md
├── test_images/
│ ├── DR/
│ └── No_DR/
├── requirements.txt
└── README.md


## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/diabetic_retinopathy_ml.git
cd diabetic_retinopathy_ml
```

``` bash
pip install -r requirements.txt
```

```bash
streamlit run pages/app.py
```

---

**Image Credit**
Sample retinal image used in documentation sourced from:
Thumbay University Hospital – Diabetic Retinopathy

---

**Author**
Sheryll Dumapal
Berlin, 2025