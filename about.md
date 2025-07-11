# Diabetic Retinopathy Prediction

### What is Diabetic Retinopathy?
Diabetic Retinopathy (DR) is a progressive eye condition caused by damage to the blood vessels in the retina — the light-sensitive layer at the back of the eye — due to prolonged high blood sugar levels. It is one of the leading causes of vision loss and blindness among working-age adults globally.

DR affects individuals with Type 1 or Type 2 diabetes, and its progression can be silent in early stages, making regular screening and early detection crucial.

### What Causes It?
The main cause of diabetic retinopathy is chronic hyperglycaemia (high blood sugar), which weakens and damages the retinal capillaries over time. As the damage progresses, blood vessels may:

- Leak fluid or blood
- Become blocked
- Stimulate abnormal vessel growth (neovascularisation), leading to scarring or retinal detachment

Several risk factors increase the likelihood of developing DR:
- Long duration of diabetes
- Poor blood sugar control
- High blood pressure
- High cholesterol
- Smoking
- Pregnancy

### Is It Preventable?
While diabetic retinopathy may not be entirely preventable, its onset and progression can often be delayed or minimised through:

- Maintaining good glycaemic control
- Managing blood pressure and cholesterol
- Quitting smoking
- Attending regular eye exams, ideally annually

### Prognosis and Diagnosis
Prognosis varies depending on how early the disease is detected and managed. In early stages, vision can often be preserved with timely treatment and lifestyle interventions. However, in advanced stages, complications like macular oedema or retinal detachment may result in irreversible vision loss.

Diagnosis typically involves a dilated eye exam by an ophthalmologist and may include:
- Retinal photography
- Optical Coherence Tomography (OCT)
- Fluorescein angiography

### Why Early Detection Matters
Early-stage DR often has no visible symptoms, but damage may already be occurring. If left undiagnosed, it can silently progress to a stage where treatment is more complex and less effective. That’s why early detection is key — it enables prompt interventions that can prevent severe outcomes.

### How Machine Learning Can Help
Machine learning (ML), particularly deep learning and Convolutional Neural Networks (CNNs), has emerged as a powerful tool in the early detection of diabetic retinopathy. By training models on large datasets of retinal images, machines can learn to:
- Detect subtle patterns of DR not easily visible to the human eye
- Provide rapid, scalable screening tools in clinical and low-resource settings
- Serve as a decision-support tool for ophthalmologists and healthcare providers
Machine learning models can augment clinical workflows by improving detection speed, accuracy, and accessibility.

### About This Machine Learning Project
This project was developed by Sheryll Dumapal to demonstrate how machine learning can be used to support diabetic retinopathy screening and diagnosis through image classification.

### Model Overview
The project began with a highly imbalanced dataset and set out to perform binary image classification to distinguish between Diabetic Retinopathy (DR) and healthy retinal images (No_DR) using Convolutional Neural Networks (CNNs). Initial experiments employed the EfficientNet family of models, known for their performance in image-based tasks. Despite incorporating strategies such as data augmentation, class weighting, undersampling, and image preprocessing, the model's performance remained limited due to the skewed class distribution.

To improve stability and reduce bias, a more balanced dataset from Kaggle was selected, prompting the development of a custom CNN architecture tailored specifically for the classification task.

#### Final Model and Dataset
A custom CNN was trained on the new dataset, which presented a more balanced ratio of DR and No_DR samples. The architecture included multiple convolutional and pooling layers, followed by dense layers and a sigmoid activation function for binary classification.

To enhance generalisation and reduce overfitting, several techniques were applied during the training pipeline:
- Data Augmentation: Random rotations, flips, and zooms were introduced using ImageDataGenerator to increase training variability.
- Image Preprocessing: Applied colour enhancement using OpenCV to improve image contrast and highlight retinal features.

**Class Imbalance Handling:**
- Undersampling of the majority class
- Class weights applied during training
- Synthetic variation through augmentation

**Training Strategy:**
- All images resized to 224x224 and normalised
- Binary cross-entropy loss used for optimisation
- Accuracy and AUC tracked as evaluation metrics
- Early stopping implemented to prevent overfitting and retain the best-performing model

#### Performance
After these adjustments, the model achieved:

**Accuracy: 94%**

Precision (DR):      0.95
Recall (DR):         0.93
F1-Score (DR):       0.94
Overall Accuracy:    94%

Precision & Recall: Balanced across both DR and No_DR classes

This marks a dramatic improvement from the initial 60% accuracy seen with the imbalanced dataset.

### Why Machine Learning Matters
- Early detection of diabetic retinopathy can prevent irreversible vision loss.
- Manual diagnosis is time-consuming and prone to variability.
- Machine learning offers a fast, consistent, and scalable way to assist medical professionals.

This approach enables pre-screening, helping reduce the burden on ophthalmologists and improve early intervention rates.

### Future Improvements
Potential extensions of the model include:
- Training with larger and more diverse datasets to improve robustness.
- Transitioning from binary to multi-class classification, identifying DR severity levels (mild, moderate, severe, proliferative).
- Integrating the model into mobile or telehealth platforms for field diagnostics.

Applying explainable AI techniques like Grad-CAM to visualise decision-making.

### Future Directions
While this model performs well for binary classification, diabetic retinopathy is a multi-stage condition. Future development could include:

### Multi-class classification across DR severity levels (e.g., mild, moderate, severe, proliferative)

- Integration of Optical Coherence Tomography (OCT) data for 3D insights
- Real-time mobile app deployment for broader accessibility
- Explainable AI tools like Grad-CAM to visualise model decisions
- Incorporation of clinical metadata (e.g., blood sugar levels, blood pressure) for more holistic predictions

Ultimately, machine learning in diabetic retinopathy holds promise for scaling early detection globally, especially in under-resourced regions, and supporting ophthalmologists with enhanced diagnostic capabilities.
