# File Name: symptom_runner.py (RAG Fixes Applied)

import pandas as pd
import joblib
import torch
import numpy as np
import os
import sys
import re 
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
from langdetect import detect 
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer, util 

# --- 1. MODEL AND DATA INITIALIZATION (Streamlined) ---
MODEL_DIR = './biobert_saved_model'
METADATA_PATH = 'biobert_model_weighted.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ... [Other model component variables remain here]

# RAG Components
RAG_MODEL = None
FAQ_EMBEDDINGS = None
FAQ_QUESTIONS = []
faq_df = None

# GLOBAL TRANSLATOR MODELS
HI_TO_EN_MODEL = None
MUL_TO_EN_MODEL = None
MUL_TO_EN_TOKENIZER = None
CONVERSATION_HISTORY = [] 

try:
    # --- LOAD CORE ML AND DATA (Assuming success) ---
    metadata = joblib.load(METADATA_PATH)
    label_encoder = metadata['label_encoder']
    MAX_LEN = metadata['max_len']
    disease_classes = metadata['classes']
    NUM_LABELS = len(disease_classes)

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR, 
        num_labels=NUM_LABELS
    ).to(DEVICE).eval() 

    description_df = pd.read_csv("symptom_Description.csv")
    precaution_df = pd.read_csv("symptom_precaution.csv")
    faq_df = pd.read_csv("faq_knowledgebase.csv") 
    
    # --- LOAD RAG COMPONENTS & PRE-CALCULATE EMBEDDINGS ---
    RAG_MODEL = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
    FAQ_QUESTIONS = faq_df['Question'].tolist()
    FAQ_EMBEDDINGS = RAG_MODEL.encode(FAQ_QUESTIONS, convert_to_tensor=True)
    
    # --- LOAD INPUT TRANSLATION MODELS ONCE ---
    HI_TO_EN_MODEL = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-hi-en').to(DEVICE)
    MUL_TO_EN_MODEL_ID = 'Helsinki-NLP/opus-mt-mul-en'
    MUL_TO_EN_MODEL = MarianMTModel.from_pretrained(MUL_TO_EN_MODEL_ID).to(DEVICE)
    MUL_TO_EN_TOKENIZER = MarianTokenizer.from_pretrained(MUL_TO_EN_MODEL_ID)
    
    print("All models and RAG knowledge base loaded successfully.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Data file not found. Ensure faq_knowledgebase.csv is in the src folder. Missing: {e}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during initialization: {e}")
    sys.exit(1)


# --- 2. CORE HELPER FUNCTIONS ---

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def final_translation(text, dest_lang):
    """Aggressively cleans source text for better MarianMT output."""
    if dest_lang == 'en':
        return text

    # AGGRESSIVE CLEANUP: (Required for stable MarianMT output)
    text = text.replace("_", " ")
    text = re.sub(r'(\s)Suggestion\s\d\s', r'\1Suggestion ', text) 
    text = re.sub(r'\(Prob:\s[0-9.]+\)', '', text) 
    text = re.sub(r'Precautions:\s*\d\.', 'Precautions:', text) 
    text = re.sub(r'[—]', '-', text) 
    text = re.sub(r'\s{2,}', ' ', text).strip() 

    # MarianMT model ID selection
    if dest_lang in ['ta', 'ml', 'kn', 'te']: 
        model_id = "Helsinki-NLP/opus-mt-en-dra"
        target_token = '>>' + ('tam' if dest_lang == 'ta' else 'mal' if dest_lang == 'kn' else 'tel' if dest_lang == 'te' else 'kan') + '<< '
        marian_text = target_token + text
    
    elif dest_lang == 'hi':
        model_id = "Helsinki-NLP/opus-mt-en-hi"
        marian_text = text
        
    else:
        return text

    try:
        mt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_mt = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        input_ids = mt_tokenizer(marian_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        translated_ids = model_mt.generate(input_ids)
        translated_text = mt_tokenizer.decode(translated_ids.squeeze(), skip_special_tokens=True)
        
        del model_mt
        del mt_tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return translated_text
        
    except Exception as e:
        return text

def translate_to_english_fallback(text):
    """
    FIX: Uses the general multilingual model for vague questions to avoid poor disease-specific translation.
    """
    try:
        src_lang = detect_language(text)
        
        # Use the general multilingual model for better quality general questions
        model_mt = MUL_TO_EN_MODEL
        mt_tokenizer = MUL_TO_EN_TOKENIZER
        
        input_text = f">>{src_lang}<< {text}" # Use lang token for multilingual input

        input_ids = mt_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        translated_ids = model_mt.generate(input_ids)
        english_text = mt_tokenizer.decode(translated_ids.squeeze(), skip_special_tokens=True)

        english_text = re.sub(r'[^a-zA-Z0-9\s,.]', '', english_text).strip()

        return english_text
    except Exception as e:
        return text

# --- RAG RETRIEVAL LOGIC (FAQ Focus) ---
def retrieve_knowledge(user_input):
    """
    Retrieves the most semantically similar FAQ answer using RAG.
    """
    input_lower = user_input.lower()
    
    # 1. First Aid is DISABLED

    # 2. Add an explicit keyword check to force RAG redirection for non-symptom queries
    non_symptom_keywords = ['how long', 'safe to take', 'dose', 'what is', 'is it safe']
    should_be_faq = any(k in input_lower for k in non_symptom_keywords)
    
    # 3. FAQ Semantic Search (RAG)
    user_embedding = RAG_MODEL.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, FAQ_EMBEDDINGS)[0]
    
    best_match_index = torch.argmax(cosine_scores).item()
    confidence = cosine_scores[best_match_index].item()
    
    # FIX: Lower the threshold to 0.65, OR trigger if a non-symptom keyword is present
    if confidence > 0.50 or (confidence > 0.25 and should_be_faq):
        question = FAQ_QUESTIONS[best_match_index]
        answer = faq_df.loc[faq_df['Question'] == question, 'Answer'].iloc[0]
        return f"FAQ Answer (Confidence: {confidence:.2f}): {answer}"

    return None # No confident RAG retrieval match


def predict_top_diseases_biobert(user_english_text, top_k=2):
    """Predicts top 2 diseases using the fine-tuned BioBERT model."""
    encoding = tokenizer.encode_plus(
        user_english_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy().flatten()

    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_results = []

    for idx in top_k_indices:
        disease_name = label_encoder.inverse_transform([idx])[0]
        prob = probabilities[idx]
        description, precautions = get_disease_info(disease_name)
        
        top_k_results.append({
            'disease': disease_name.title(),
            'probability': prob,
            'description': description,
            'precautions': precautions
        })
        
    return top_k_results

def get_disease_info(disease_name):
    """Retrieves English description and precautions."""
    disease_name = disease_name.strip().lower()
    
    desc_row = description_df[description_df['Disease'].str.lower().str.strip() == disease_name]
    description = desc_row.iloc[0]['Description'].strip() if not desc_row.empty else "Description not available."
    
    prec_row = precaution_df[precaution_df['Disease'].str.lower().str.strip() == disease_name]
    precaution_list = [
        prec_row.iloc[0][f'Precaution_{i}'].strip()
        for i in range(1, 5) 
        if not pd.isna(prec_row.iloc[0].get(f'Precaution_{i}'))
    ] if not prec_row.empty else ["Consult a doctor immediately if symptoms worsen."]
    return description, precaution_list


# --- 3. MAIN CHATBOT LOGIC ---

def run_chatbot(user_input):
    global CONVERSATION_HISTORY
    
    source_lang = detect_language(user_input)
    
    if not user_input.strip():
        english_error = "Please tell me how you are feeling."
        return {'english': english_error, 'translated': final_translation(english_error, source_lang), 'language': source_lang, 'mode': 'Initial'}
        
    
    print(f"\n--- Processing Input: '{user_input}' (Detected Language: {source_lang}) ---")

    # STEP 0: Check for RAG/Knowledge Base intent (FAQ)
    kb_response_en = retrieve_knowledge(user_input)
    if kb_response_en:
        CONVERSATION_HISTORY.append({'user': user_input, 'mode': 'KB'})
        return {
            'english': kb_response_en,
            'translated': final_translation(kb_response_en, source_lang),
            'language': source_lang,
            'mode': 'Knowledge Base (FAQ)'
        }

    # STEP 1: TRANSLATE INPUT TO ENGLISH (for BioBERT)
    if source_lang != 'en':
        english_input = translate_to_english_fallback(user_input)
        print(f"Translated English Input: {english_input}")
    else:
        english_input = user_input
    
    # --- STEP 2: BIOBERT PREDICTION ---
    
    top_results = predict_top_diseases_biobert(english_input, top_k=2)
    
    if not top_results or top_results[0]['probability'] < 0.05: 
        english_error = "I could not find a confident diagnosis based on your input. The condition might be too vague or complex."
        CONVERSATION_HISTORY.append({'user': user_input, 'mode': 'Error'})
        return {
            'english': english_error,
            'translated': final_translation(english_error, source_lang),
            'language': source_lang,
            'mode': 'Error'
        }

    # STEP 3: CONSTRUCT CLEAN ENGLISH RESPONSE 
    english_response = f"Based on the symptoms: '{english_input}', here are the top 2 possible conditions:\n\n"
    
    # Personalized Experience (Basic)
    if CONVERSATION_HISTORY and CONVERSATION_HISTORY[-1]['mode'] == 'Symptom Checker':
        english_response = "Following up on your last symptoms, " + english_response
    
    for i, result in enumerate(top_results):
        english_response += f"Suggestion {i+1} (Prob: {result['probability']:.2f}): {result['disease']}\n"
        english_response += f"Description: {result['description']}\n"
        english_response += "Precautions:\n"
        for j, p in enumerate(result['precautions']):
            english_response += f"  {j+1}. {p}\n"
        english_response += "\n"

    CONVERSATION_HISTORY.append({'user': user_input, 'mode': 'Symptom Checker', 'disease': top_results[0]['disease']})

    # STEP 4: FINAL TRANSLATION 
    final_response_translated = final_translation(english_response, source_lang)
    
    return {
        'english': english_response,
        'translated': final_response_translated,
        'language': source_lang,
        'mode': 'BioBERT Classifier'
    }

# --- 4. EXAMPLE USAGE ---

def format_output(result):
    if isinstance(result, str):
        return result
    
    print(f"\n--- Chatbot Mode: {result.get('mode', 'N/A')} (Language: {result['language'].upper()}) ---")
    print("\n--- Final Translated Response ---")
    print(result['translated'])
    print("\n--- English Debugging Output ---")
    print(result['english'])

if __name__ == '__main__':
    # You must have run biobert_trainer.py and created faq_knowledgebase.csv before this!
    
    print("=======================================================")
    print("TEST 1: FAQ Semantic Search (RAG) - FIX Check")
    print("=======================================================")
    # This should trigger the RAG retrieval
    test_input_1 = "Is it safe to take ibuprofen for my pain?"
    format_output(run_chatbot(test_input_1))
    
    print("\n=======================================================")
    print("TEST 2: Symptom Check (English)")
    print("=======================================================")
    # This should trigger the BioBERT classifier
    test_input_2 = "I have a severe headache and feel very dizzy." 
    format_output(run_chatbot(test_input_2))
    
    print("\n=======================================================")
    print("TEST 3: RAG Semantic Search in Hindi - Translation Fix")
    print("=======================================================")
    # This should trigger the RAG retrieval (Translation check)
    hindi_input_3 = "सर्दी जुकाम कितने दिन तक रहता है?" # How long does a cold last?
    format_output(run_chatbot(hindi_input_3))