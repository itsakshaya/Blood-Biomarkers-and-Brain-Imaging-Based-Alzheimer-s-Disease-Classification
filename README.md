# Alzheimer's Disease Classification System

## Blood Biomarkers and Brain Imaging Based Alzheimer's Disease Classification

A machine learning-based Streamlit application for classifying Alzheimer's disease using clinical information and brain MRI images.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- OpenCV
- Random Forest
- Streamlit
- Joblib
- ReportLab

## Features

- Clinical data classification
- MRI image classification
- Classification confidence scores
- Interactive Streamlit application
- PDF classification report generation

## Machine Learning

The project uses Random Forest classification models.

### Clinical Model

The clinical model uses patient information such as:

- Age
- MMSE Score
- CDR Score
- Memory Score
- Gender

### MRI Model

MRI images are:

1. Resized to 64 × 64 pixels
2. Converted to grayscale
3. Converted into numerical features
4. Classified using a Random Forest model

## Project Structure

```text
├── app.py
├── clinical_model.pkl
├── clinical_scaler.pkl
├── mri_model.pkl
├── mri_label_encoder.pkl
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

### 3. Open the application

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 4. Use the Application

1. Enter the required patient details.
2. Upload an MRI image.
3. Click **Run Classification**.
4. View the classification results and confidence scores.
5. Download the generated PDF report.
