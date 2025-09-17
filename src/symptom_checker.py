import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data_path = os.path.join(os.path.dirname(__file__), "../data/")
symptoms_df = pd.read_csv(os.path.join(data_path, "Final_Augmented_dataset_Diseases_and_Symptoms.csv"))
symptoms_long = symptoms_df.melt(id_vars=["diseases"], var_name="symptom", value_name="has_symptom")
symptoms_long = symptoms_long[symptoms_long["has_symptom"].notna() & (symptoms_long["has_symptom"] != 0)]
def clean_text(text):
    if pd.isna(text):
        return None
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text
symptoms_long["disease"] = symptoms_long["diseases"].apply(clean_text)
symptoms_long["symptom"] = symptoms_long["symptom"].apply(clean_text)
symptom_df = symptoms_long[["symptom", "disease"]]
X_train, X_test, y_train, y_test = train_test_split(symptom_df["symptom"], symptom_df["disease"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(clf, os.path.join(data_path, "symptom_model.pkl"))
joblib.dump(vectorizer, os.path.join(data_path, "symptom_vectorizer.pkl"))
