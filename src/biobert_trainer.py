# File Name: biobert_trainer.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# --- Configuration ---
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.2' 
TRAINING_FILE = "symptoms_text_data.csv" 
SAVE_PATH = "biobert_model_weighted.pkl"
BATCH_SIZE = 8
EPOCHS = 4 # Increased epochs for high-quality data
MAX_LEN = 128 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Data Loading and Preprocessing ---
try:
    df = pd.read_csv(TRAINING_FILE)
    if df.empty:
        raise ValueError("Training file is empty.")
except Exception as e:
    print(f"CRITICAL ERROR: Cannot load or process training data. {e}")
    sys.exit(1)

# Encode disease labels to integers (0, 1, 2, ...)
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
NUM_LABELS = len(label_encoder.classes_)
print(f"Total diseases (labels): {NUM_LABELS}")

# Split data (using 20% test_size now that we have more diverse data)
try:
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label_id']
    )
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
except ValueError as e:
    # Fallback to a larger, safer validation size if 20% is too small for stratification
    print(f"Stratification Error: {e}")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label_id']
    )


# --- 2. Custom Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 3. Model and Tokenizer Initialization ---
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS
).to(DEVICE)

train_dataset = TextDataset(train_df['text'], train_df['label_id'], tokenizer, MAX_LEN)
val_dataset = TextDataset(val_df['text'], val_df['label_id'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 4. Training Loop ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_predictions / len(val_df)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


# --- 5. Save the Fine-Tuned Model Components ---
model.save_pretrained('./biobert_saved_model')
tokenizer.save_pretrained('./biobert_saved_model')

joblib.dump({
    'label_encoder': label_encoder,
    'tokenizer_name': MODEL_NAME,
    'max_len': MAX_LEN,
    'classes': label_encoder.classes_.tolist()
}, SAVE_PATH)

print(f"\nBioBERT model, tokenizer, and encoder saved successfully to 'biobert_saved_model' directory and '{SAVE_PATH}'")