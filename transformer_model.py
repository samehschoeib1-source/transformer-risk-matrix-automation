%%writefile transformer_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# --- 1. PyTorch Dataset Wrapper ---
class PHMSAIncidentDataset(Dataset):
    def __init__(self, texts, impact_labels, likelihood_labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.impact_labels = impact_labels.tolist()
        self.likelihood_labels = likelihood_labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'impact_label': torch.tensor(self.impact_labels[idx], dtype=torch.long),
            'likelihood_label': torch.tensor(self.likelihood_labels[idx], dtype=torch.long)
        }

# --- 2. Multi-Task Transformer Model Architecture ---
class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_impact_classes=4, num_likelihood_classes=3):
        super(MultiTaskTransformer, self).__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        self.dropout = nn.Dropout(0.2)
        
        # Dual Classification Heads (Y-Axis & X-Axis)
        self.impact_head = nn.Linear(hidden_size, num_impact_classes)
        self.likelihood_head = nn.Linear(hidden_size, num_likelihood_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token vector
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        
        # Concurrent predictions
        impact_logits = self.impact_head(cls_rep)
        likelihood_logits = self.likelihood_head(cls_rep)
        
        return impact_logits, likelihood_logits

# Helper function to compute aligned class weights safely
def get_aligned_class_weights(y_train, num_total_classes):
    unique_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    
    # Initialize full weight tensor with 1.0
    full_weights = np.ones(num_total_classes, dtype=np.float32)
    for c, w in zip(unique_classes, weights):
        if c < num_total_classes:
            full_weights[int(c)] = w
            
    return full_weights

# --- 3. Training & Evaluation Pipeline ---
def train_multi_task_model(model_name='distilbert-base-uncased', epochs=3, batch_size=16, lr=2e-5):
    input_file = 'engineered_risk_data.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    print(f"Loading base Transformer encoder: {model_name}...")

    try:
        df = pd.read_csv(input_file, encoding='utf-8').dropna(subset=['NARRATIVE'])
        
        X = df['NARRATIVE']
        y_impact = df['IMPACT_SEVERITY']
        y_likelihood = df['ESCALATION_LIKELIHOOD']

        # Train/Test Split (80/20) matching academic standard
        X_train, X_test, y_imp_tr, y_imp_te, y_lik_tr, y_lik_te = train_test_split(
            X, y_impact, y_likelihood, test_size=0.20, random_state=42
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        train_dataset = PHMSAIncidentDataset(X_train, y_imp_tr, y_lik_tr, tokenizer)
        test_dataset = PHMSAIncidentDataset(X_test, y_imp_te, y_lik_te, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # FIXED: Robust class weight calculation preventing ValueError
        imp_weights = get_aligned_class_weights(y_imp_tr, num_total_classes=4)
        lik_weights = get_aligned_class_weights(y_lik_tr, num_total_classes=3)
        
        criterion_impact = nn.CrossEntropyLoss(weight=torch.tensor(imp_weights, dtype=torch.float).to(device))
        criterion_likelihood = nn.CrossEntropyLoss(weight=torch.tensor(lik_weights, dtype=torch.float).to(device))

        model = MultiTaskTransformer(model_name=model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # --- Training Loop ---
        print("\n--- Starting Multi-Task Fine-Tuning ---")
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_impact = batch['impact_label'].to(device)
                labels_likelihood = batch['likelihood_label'].to(device)
                
                logits_impact, logits_likelihood = model(input_ids, attention_mask)
                
                loss_imp = criterion_impact(logits_impact, labels_impact)
                loss_lik = criterion_likelihood(logits_likelihood, labels_likelihood)
                
                # Joint Loss Optimization
                total_loss = loss_imp + loss_lik
                
                total_loss.backward()
                optimizer.step()
                
                total_train_loss += total_loss.item()
                
            avg_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

        # --- Evaluation Phase ---
        print("\n================ TRANSFORMER EVALUATION REPORT ================")
        model.eval()
        imp_preds, imp_true = [], []
        lik_preds, lik_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits_impact, logits_likelihood = model(input_ids, attention_mask)
                
                imp_preds.extend(torch.argmax(logits_impact, dim=1).cpu().numpy())
                imp_true.extend(batch['impact_label'].numpy())
                
                lik_preds.extend(torch.argmax(logits_likelihood, dim=1).cpu().numpy())
                lik_true.extend(batch['likelihood_label'].numpy())

        print(f"Impact Severity Accuracy: {accuracy_score(imp_true, imp_preds):.4f}")
        print("\nImpact Severity Classification Metrics:")
        print(classification_report(imp_true, imp_preds, zero_division=0))
        
        print("-" * 60)
        print(f"Escalation Likelihood Accuracy: {accuracy_score(lik_true, lik_preds):.4f}")
        print("\nEscalation Likelihood Classification Metrics:")
        print(classification_report(lik_true, lik_preds, zero_division=0))
        print("==============================================================")

    except FileNotFoundError:
        print(f"Error: Required file '{input_file}' not found. Ensure engineer_labels.py has been run.")

if __name__ == "__main__":
    train_multi_task_model(model_name='distilbert-base-uncased', epochs=3)
