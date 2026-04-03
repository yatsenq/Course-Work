import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from pathlib import Path

# Конфігурація шляхів та параметрів
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS = {
    'Head A': 'Models/DETECTOR/BERT/head_a_best.pt',
    'Head B': 'Models/DETECTOR/BERT/head_b_best.pt',
    'Head C': 'Models/DETECTOR/BERT/head_c_best.pt'
}
TOKENIZER_PATH = 'Models/DETECTOR/BERT/bert_tokenizer'
DATASET_PATH = 'Datasets/news_detector/last_dataset.csv'
RESULTS_FILE = 'Models/DETECTOR/BERT/comparison_results.csv'

# Оголошення архітектур
class BertClassifierHeadA(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))

class BertClassifierHeadB(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, attention_mask):
        scores = self.attention(hidden_states).squeeze(-1).masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

class BertClassifierHeadC(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.attention_pool = AttentionPooling(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.attention_pool(outputs.last_hidden_state, attention_mask))

HEAD_CLASSES = {'Head A': BertClassifierHeadA, 'Head B': BertClassifierHeadB, 'Head C': BertClassifierHeadC}

def evaluate_model(head_name, checkpoint_file, test_df, tokenizer, max_len):
    print(f"\n--- Оцінювання: {head_name} ---")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model_name = checkpoint['model_name']
    
    ModelClass = HEAD_CLASSES[head_name]
    model = ModelClass(model_name).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    y_true = test_df['label'].values
    y_pred = []
    
    with torch.no_grad():
        for text in test_df['text']:
            encoding = tokenizer(text, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
            ids, mask = encoding['input_ids'].to(DEVICE), encoding['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            y_pred.append(torch.argmax(logits, dim=1).item())
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1

def main():
    df = pd.read_csv(DATASET_PATH).dropna(subset=['text', 'label'])
    _, test_df = pd.concat([df[df.label==0].sample(frac=0.2), df[df.label==1].sample(frac=0.2)]), None # Sample for quick comparison
    # Actually just split properly
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    results = []
    for head_name, path in CHECKPOINTS.items():
        if os.path.exists(path):
            acc, f1 = evaluate_model(head_name, path, test_df, tokenizer, 256)
            results.append({'Model': head_name, 'Accuracy': acc, 'F1-Score': f1})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nРезультати збережено в {RESULTS_FILE}")

if __name__ == "__main__":
    main()
