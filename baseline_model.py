import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

def run_baseline_pipeline(input_file='engineered_risk_data.csv'):
    print("--- Step 5: Training ML Baseline Model (Balanced Training Split) ---")
    print(f"Loading engineered risk dataset: {input_file}...")
    
    # Load dataset with low_memory=False to suppress mixed type warnings
    df = pd.read_csv(input_file, encoding='utf-8', low_memory=False).dropna(subset=['NARRATIVE'])
    
    X = df['NARRATIVE'].values
    y_impact = df['IMPACT_SEVERITY'].values
    y_escalation = df['ESCALATION_LIKELIHOOD'].values

    # Stratified 80/20 Split on Impact Severity
    X_train, X_test, y_train_imp, y_test_imp, y_train_esc, y_test_esc = train_test_split(
        X, y_impact, y_escalation, test_size=0.2, random_state=42, stratify=y_impact
    )

    print(f"Original Training observations: {len(X_train)} | Test observations: {len(X_test)}")

    # Vectorize Text Data via TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Balance Training Set via Oversampling
    ros_imp = RandomOverSampler(random_state=42)
    X_train_imp_res, y_train_imp_res = ros_imp.fit_resample(X_train_tfidf, y_train_imp)

    ros_esc = RandomOverSampler(random_state=42)
    X_train_esc_res, y_train_esc_res = ros_esc.fit_resample(X_train_tfidf, y_train_esc)

    print(f"Balanced Training observations (Impact): {X_train_imp_res.shape[0]}")
    print(f"Balanced Training observations (Likelihood): {X_train_esc_res.shape[0]}")

    # --- 1. Train & Predict Impact Severity ---
    rf_impact = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_impact.fit(X_train_imp_res, y_train_imp_res)
    y_pred_imp = rf_impact.predict(X_test_tfidf)

    # --- 2. Train & Predict Escalation Likelihood ---
    rf_escalation = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_escalation.fit(X_train_esc_res, y_train_esc_res)
    y_pred_esc = rf_escalation.predict(X_test_tfidf)

    # --- Print Evaluation Report ---
    imp_acc = accuracy_score(y_test_imp, y_pred_imp)
    esc_acc = accuracy_score(y_test_esc, y_pred_esc)

    print("\n================ BASELINE (BALANCED) EVALUATION REPORT ================")
    print(f"Impact Severity Target Accuracy: {imp_acc:.4f}\n")
    print("Impact Classification Metrics:")
    print(classification_report(y_test_imp, y_pred_imp))
    
    print("-" * 60)
    print(f"Escalation Likelihood Target Accuracy: {esc_acc:.4f}\n")
    print("Likelihood Classification Metrics:")
    print(classification_report(y_test_esc, y_pred_esc))
    print("=======================================================================")

    # --- Export Results for Step 7 (evaluate_comparison.py) ---
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(
        'results/baseline_preds.npz',
        y_true_impact=y_test_imp,
        y_pred_impact=y_pred_imp,
        y_true_escalation=y_test_esc,
        y_pred_escalation=y_pred_esc
    )
    print("\n[SUCCESS] Baseline test predictions exported to 'results/baseline_preds.npz'")

if __name__ == "__main__":
    run_baseline_pipeline()
