# Liver Patient Prediction Using Machine Learning

## Project Overview

This project predicts whether a patient is likely to have liver disease based on demographic and clinical laboratory measurements. Multiple machine learning algorithms are trained and evaluated, and the model with the highest F1 Score is selected as the final model.

## Dataset

The project uses the **Indian Liver Patient Dataset** containing patient information and medical measurements.

### Features

* Age
* Gender
* Total Bilirubin
* Direct Bilirubin
* Alkaline Phosphotase
* Alamine Aminotransferase
* Aspartate Aminotransferase
* Total Proteins
* Albumin
* Albumin and Globulin Ratio

### Target

* `1` → Liver Disease
* `2` → No Liver Disease

## Data Preprocessing

* Missing values were handled using median imputation.
* Gender was encoded into numerical values.
* The dataset was divided into training and testing sets.
* StandardScaler was applied for models that require feature scaling.

## Machine Learning Models

The following algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The model with the highest **F1 Score** was selected as the final model.

## Deployment

The trained machine learning workflow is deployed using Streamlit.

The application allows the user to enter patient details and receive a prediction:

* Liver Disease Predicted
* No Liver Disease Predicted


## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Jupyter Notebook

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Limitations

* The dataset is limited in size.
* Class imbalance may affect model performance.
* Model performance can vary depending on the data split.
* The application is intended for academic and educational purposes and should not replace professional medical diagnosis.
