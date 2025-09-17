import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

dataset_df = pd.read_csv("dataset.csv")  
description_df = pd.read_csv("symptom_Description.csv")  

symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]  # Symptom_1 to Symptom_17
dataset_df['Symptoms_List'] = dataset_df[symptom_cols].apply(
    lambda x: [str(s).strip().lower() for s in x if pd.notna(s)],
    axis=1
)

dataset_df['Disease'] = dataset_df['Disease'].str.lower().str.strip()  # ensure lowercase

all_symptoms = sorted(list({symptom for sublist in dataset_df['Symptoms_List'] for symptom in sublist}))
print(f"Number of unique symptoms: {len(all_symptoms)}")

# -----------------------------
# 5. Multi-hot encode symptoms
# -----------------------------
for symptom in all_symptoms:
    dataset_df[symptom] = dataset_df['Symptoms_List'].apply(lambda x: 1 if symptom in x else 0)

# -----------------------------
# 6. Prepare features and labels
# -----------------------------
X = dataset_df[all_symptoms]
y = dataset_df['Disease']

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 8. Train Random Forest Classifier
# -----------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# 9. Evaluate model
# -----------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# -----------------------------
# 10. Save the trained model
# -----------------------------
joblib.dump(clf, "symptom_checker_model.pkl")
print("Trained model saved as 'symptom_checker_model.pkl'")
