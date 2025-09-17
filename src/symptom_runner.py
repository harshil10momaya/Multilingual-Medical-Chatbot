import pandas as pd
import joblib

# -----------------------------
# Load saved model and description
# -----------------------------
model = joblib.load("symptom_checker_model.pkl")
description_df = pd.read_csv("symptom_Description.csv")  # Disease + Description

# Load the symptom vocabulary (the same as training)
dataset_df = pd.read_csv("dataset.csv")
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
dataset_df['Symptoms_List'] = dataset_df[symptom_cols].apply(
    lambda x: [str(s).strip().lower() for s in x if pd.notna(s)],
    axis=1
)
all_symptoms = sorted(list({symptom for sublist in dataset_df['Symptoms_List'] for symptom in sublist}))

# -----------------------------
# Prediction function
# -----------------------------
def predict_disease(user_symptoms):
    """
    user_symptoms: list of symptom strings, e.g., ["fever", "cough"]
    returns: predicted disease and its description
    """
    # 1. Convert user symptoms to multi-hot vector
    input_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)
    
    # 2. Predict disease
    predicted_disease = model.predict(input_df)[0]
    
    # 3. Get disease description
    description_row = description_df[description_df['Disease'].str.lower() == predicted_disease]
    if not description_row.empty:
        disease_description = description_row.iloc[0]['Description']
    else:
        disease_description = "Description not available."
    
    return predicted_disease, disease_description

# -----------------------------
# Example usage
# -----------------------------
user_symptoms = ["fever", "cough"]
disease, description = predict_disease(user_symptoms)
print(f"Predicted Disease: {disease}")
print(f"Description: {description}")