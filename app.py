import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# PAGE TITLE
# ==========================================

st.set_page_config(
    page_title="Liver Patient Prediction",
    page_icon="🩺"
)

st.title("🩺 Liver Patient Prediction")
st.write(
    "Enter the patient details to predict "
    "whether the patient is likely to have liver disease."
)


# ==========================================
# LOAD DATASET
# ==========================================

columns = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Target"
]

df = pd.read_csv(
    "Liver predicition.csv",
    header=None,
    names=columns
)


# ==========================================
# DATA PREPROCESSING
# ==========================================

# Clean Gender
df["Gender"] = df["Gender"].astype(str).str.strip()

# Encode Gender
df["Gender"] = df["Gender"].replace({
    "Male": 1,
    "Female": 0
})

# Fill missing values
df["Albumin_and_Globulin_Ratio"] = (
    df["Albumin_and_Globulin_Ratio"]
    .fillna(
        df["Albumin_and_Globulin_Ratio"].median()
    )
)


# ==========================================
# X AND Y
# ==========================================

X = df.drop("Target", axis=1)

y = df["Target"].map({
    1: 1,
    2: 0
})


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# STANDARD SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ==========================================
# TRAIN ALL MODELS
# ==========================================

for name, model in models.items():

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        model.fit(
            X_train_scaled,
            y_train
        )

    else:

        model.fit(
            X_train,
            y_train
        )


# ==========================================
# MODEL EVALUATION
# ==========================================

results = []

for name, model in models.items():

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

    results.append({

        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    })


results_df = pd.DataFrame(results)


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

final_model = models[best_model_name]


# ==========================================
# DISPLAY MODEL RESULTS
# ==========================================

st.subheader("Model Performance")

st.dataframe(
    results_df.round(3)
)

st.info(
    f"Selected Final Model: {best_model_name}"
)


# ==========================================
# PATIENT INPUT
# ==========================================

st.subheader("Enter Patient Details")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


total_bilirubin = st.number_input(
    "Total Bilirubin",
    min_value=0.0,
    value=1.0
)


direct_bilirubin = st.number_input(
    "Direct Bilirubin",
    min_value=0.0,
    value=0.5
)


alkaline_phosphotase = st.number_input(
    "Alkaline Phosphotase",
    min_value=0.0,
    value=200.0
)


alamine_aminotransferase = st.number_input(
    "Alamine Aminotransferase",
    min_value=0.0,
    value=30.0
)


aspartate_aminotransferase = st.number_input(
    "Aspartate Aminotransferase",
    min_value=0.0,
    value=30.0
)


total_proteins = st.number_input(
    "Total Proteins",
    min_value=0.0,
    value=6.5
)


albumin = st.number_input(
    "Albumin",
    min_value=0.0,
    value=3.5
)


albumin_globulin_ratio = st.number_input(
    "Albumin and Globulin Ratio",
    min_value=0.0,
    value=1.0
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("🔍 Predict"):

    # Convert gender
    gender_value = 1 if gender == "Male" else 0


    # Create input dataframe
    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender_value],

        "Total_Bilirubin":
            [total_bilirubin],

        "Direct_Bilirubin":
            [direct_bilirubin],

        "Alkaline_Phosphotase":
            [alkaline_phosphotase],

        "Alamine_Aminotransferase":
            [alamine_aminotransferase],

        "Aspartate_Aminotransferase":
            [aspartate_aminotransferase],

        "Total_Proteins":
            [total_proteins],

        "Albumin":
            [albumin],

        "Albumin_and_Globulin_Ratio":
            [albumin_globulin_ratio]
    })


    # ======================================
    # PREDICTION
    # ======================================

    if best_model_name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        input_scaled = scaler.transform(
            input_data
        )

        prediction = final_model.predict(
            input_scaled
        )[0]

        probability = final_model.predict_proba(
            input_scaled
        )[0][1]

    else:

        prediction = final_model.predict(
            input_data
        )[0]

        probability = final_model.predict_proba(
            input_data
        )[0][1]


    # ======================================
    # RESULT
    # ======================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Liver Disease Predicted"
        )

    else:

        st.success(
            "✅ No Liver Disease Predicted"
        )


    st.write(
        f"Prediction Probability: "
        f"{probability:.2%}"
    )


    st.warning(
        "This application is for academic and "
        "educational purposes only and should "
        "not replace professional medical diagnosis."
    )