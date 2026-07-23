import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_baseline():
    input_file = 'engineered_risk_data.csv'
    print(f"Loading engineered risk dataset: {input_file}...")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # Drop rows where narrative text is missing or blank
        df = df.dropna(subset=['NARRATIVE'])
        
        X = df['NARRATIVE']
        y_impact = df['IMPACT_SEVERITY']
        y_likelihood = df['ESCALATION_LIKELIHOOD']
        
        # --- 1. Split Data into Train and Test Sets (80/20 split) ---
        # NO stratify parameter to prevent crashes with single-instance classes
        X_train, X_test, y_imp_train, y_imp_test, y_lik_train, y_lik_test = train_test_split(
            X, y_impact, y_likelihood, 
            test_size=0.20, 
            random_state=42
        )
        
        print(f"Training observations: {len(X_train)} | Test observations: {len(X_test)}")
        
        # --- 2. Feature Extraction using TF-IDF ---
        print("Vectorizing raw narrative text strings using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # --- 3. Train Classifier 1: Impact Severity ---
        print("Training Random Forest Classifier for Impact Severity (Y-Axis)...")
        rf_impact = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_impact.fit(X_train_tfidf, y_imp_train)
        preds_impact = rf_impact.predict(X_test_tfidf)
        
        # --- 4. Train Classifier 2: Escalation Likelihood ---
        print("Training Random Forest Classifier for Escalation Likelihood (X-Axis)...")
        rf_likelihood = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_likelihood.fit(X_train_tfidf, y_lik_train)
        preds_likelihood = rf_likelihood.predict(X_test_tfidf)
        
        # --- 5. Output Baseline Performance Reports ---
        print("\n================ BASELINE EVALUATION REPORT ================")
        print(f"Impact Severity Target Accuracy: {accuracy_score(y_imp_test, preds_impact):.4f}")
        print("\nImpact Classification Metrics:")
        print(classification_report(y_imp_test, preds_impact, zero_division=0))
        
        print("-" * 60)
        print(f"Escalation Likelihood Target Accuracy: {accuracy_score(y_lik_test, preds_likelihood):.4f}")
        print("\nLikelihood Classification Metrics:")
        print(classification_report(y_lik_test, preds_likelihood, zero_division=0))
        print("============================================================")
        
    except FileNotFoundError:
        print(f"Error: Required file '{input_file}' not found. Run engineer_labels.py first.")

if __name__ == "__main__":
    train_baseline()
