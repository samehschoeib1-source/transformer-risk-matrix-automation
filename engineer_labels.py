import pandas as pd
import numpy as np

def engineer_risk_targets():
    input_file = 'cleaned_incident_data.csv'
    output_file = 'engineered_risk_data.csv'
    
    print(f"Loading cleaned dataset: {input_file}...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # --- 1. Target Axis 1: Impact Severity ---
        print("Engineering Impact Severity categories...")
        conditions_severity = [
            (df['FATAL'] > 0) | (df['INJURE'] > 0), # Critical
            (df['SHUTDOWN_DUE_ACCIDENT_IND'] == 'YES'), # High
            (df['UNINTENTIONAL_RELEASE'] > df['UNINTENTIONAL_RELEASE'].median()) # Medium
        ]
        choices_severity = [3, 2, 1]
        # Default fallback is 0 (Low)
        df['IMPACT_SEVERITY'] = np.select(conditions_severity, choices_severity, default=0)
        
        # --- 2. Target Axis 2: Escalation Likelihood ---
        print("Engineering Escalation Likelihood categories...")
        # Higher risk if it ignited or happened on a highly pressurized line segment
        median_psig = df['ACCIDENT_PSIG'].median()
        conditions_likelihood = [
            (df['IGNITE_IND'] == 'YES'), # High Risk
            (df['ACCIDENT_PSIG'] > median_psig) # Medium Risk
        ]
        choices_likelihood = [2, 1]
        # Default fallback is 0 (Low Risk)
        df['ESCALATION_LIKELIHOOD'] = np.select(conditions_likelihood, choices_likelihood, default=0)
        
        # Print class distribution summaries to verify no empty slices
        print("\n=== Target Matrix Distributions ===")
        print("Impact Severity (0=Low, 3=Critical):\n", df['IMPACT_SEVERITY'].value_counts().sort_index())
        print("Escalation Likelihood (0=Low, 2=High):\n", df['ESCALATION_LIKELIHOOD'].value_counts().sort_index())
        
        # Save the dataset with its brand new target labels
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSaved newly labeled dataset as: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Required file '{input_file}' not found. Run clean_data.py first.")

if __name__ == "__main__":
    engineer_risk_targets()
