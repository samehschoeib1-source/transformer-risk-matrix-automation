import pandas as pd
import numpy as np

def run_eda():
    input_file = 'engineered_risk_data.csv'
    print(f"=== Starting Exploratory Data Analysis on {input_file} ===\n")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # 1. Dataset Shape & Completeness
        print(f"Total Rows: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        
        # 2. Text Narrative Analysis
        if 'NARRATIVE' in df.columns:
            narratives = df['NARRATIVE'].dropna().astype(str)
            word_counts = narratives.apply(lambda x: len(x.split()))
            char_counts = narratives.apply(len)
            
            print("\n--- Narrative Text Statistics ---")
            print(f"Valid Text Narratives: {len(narratives)}")
            print(f"Mean Word Count: {word_counts.mean():.1f} words")
            print(f"Median Word Count: {word_counts.median():.1f} words")
            print(f"Max Word Count: {word_counts.max()} words")
            print(f"95th Percentile Word Count: {np.percentile(word_counts, 95):.1f} words")
        
        # 3. Y-Axis Impact Severity Distribution
        print("\n--- Target 1: Impact Severity (Y-Axis) Distribution ---")
        impact_counts = df['IMPACT_SEVERITY'].value_counts().sort_index()
        impact_map = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)', 3: 'Critical (3)'}
        for level, count in impact_counts.items():
            pct = (count / len(df)) * 100
            print(f"  Class {impact_map.get(level, level)}: {count} records ({pct:.2f}%)")
            
        # 4. X-Axis Escalation Likelihood Distribution
        print("\n--- Target 2: Escalation Likelihood (X-Axis) Distribution ---")
        lik_counts = df['ESCALATION_LIKELIHOOD'].value_counts().sort_index()
        lik_map = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
        for level, count in lik_counts.items():
            pct = (count / len(df)) * 100
            print(f"  Class {lik_map.get(level, level)}: {count} records ({pct:.2f}%)")
            
        # 5. Risk Assessment Matrix (Cross-Tabulation)
        print("\n--- Dual-Axis 4x3 Risk Assessment Matrix Distribution ---")
        matrix_crosstab = pd.crosstab(
            df['IMPACT_SEVERITY'].map(impact_map), 
            df['ESCALATION_LIKELIHOOD'].map(lik_map),
            margins=True
        )
        print(matrix_crosstab)
        print("\n================ EDA COMPLETE ================")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}'. Please run engineer_labels.py first.")

if __name__ == "__main__":
    run_eda()