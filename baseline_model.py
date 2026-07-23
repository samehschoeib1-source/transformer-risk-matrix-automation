import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

def run_baseline():
    input_file = 'engineered_risk_data.csv'
    print(f"Loading engineered risk dataset: {input_file}...")

    try:
        df = pd.read_csv(input_file, encoding='utf-8').dropna(subset=['NARRATIVE'])
        
        X = df['NARRATIVE']
        y_impact = df['IMPACT_SEVERITY']
        y_likelihood = df['ESCALATION_LIKELIHOOD']

        # 1. Standard Train/Test Split (80/20) with random_state=42
        X_train, X_test, y_imp_tr, y_imp_te, y_lik_tr, y_lik_te = train_test_split(
            X, y_impact, y_likelihood, test_size=0.20, random_state=42
        )

        print(f"Original Training observations: {len(X_train)} | Test observations: {len(X_test)}")

        # 2. Oversample Training Set Only (Apples-to-Apples Balancing)
        ros_imp = RandomOverSampler(random_state=42)
        X_train_imp_res, y_imp_tr_res = ros_imp.fit_resample(X_train.to_frame(), y_imp_tr)
        
        ros_lik = RandomOverSampler(random_state=42)
        X_train_lik_res, y_lik_tr_res = ros_lik.fit_resample(X_train.to_frame(), y_lik_tr)

        print(f"Balanced Training observations (Impact): {len(X_train_imp_res)}")
        print(f"Balanced Training observations (Likelihood): {len(X_train_lik_res)}")

        # 3. TF-IDF Vectorization
        tfidf_imp = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tr_imp_vec = tfidf_imp.fit_transform(X_train_imp_res['NARRATIVE'])
        X_te_imp_vec = tfidf_imp.transform(X_test)

        tfidf_lik = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tr_lik_vec = tfidf_lik.fit_transform(X_train_lik_res['NARRATIVE'])
        X_te_lik_vec = tfidf_lik.transform(X_test)

        # 4. Train Random Forest Baseline
        rf_impact = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_impact.fit(X_tr_imp_vec, y_imp_tr_res)
        imp_preds = rf_impact.predict(X_te_imp_vec)

        rf_lik = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_lik.fit(X_tr_lik_vec, y_lik_tr_res)
        lik_preds = rf_lik.predict(X_te_lik_vec)

        # 5. Report Results
        print("\n================ BASELINE (BALANCED) EVALUATION REPORT ================")
        print(f"Impact Severity Target Accuracy: {accuracy_score(y_imp_te, imp_preds):.4f}")
        print("\nImpact Classification Metrics:")
        print(classification_report(y_imp_te, imp_preds, zero_division=0))

        print("-" * 60)
        print(f"Escalation Likelihood Target Accuracy: {accuracy_score(y_lik_te, lik_preds):.4f}")
        print("\nLikelihood Classification Metrics:")
        print(classification_report(y_lik_te, lik_preds, zero_division=0))
        print("=======================================================================")

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")

if __name__ == "__main__":
    run_baseline()
