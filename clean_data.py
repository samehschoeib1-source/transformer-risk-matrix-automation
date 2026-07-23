import pandas as pd
import numpy as np

def clean_data():
    input_file = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv'
    output_file = 'cleaned_incident_data.csv'
    
    print(f"Loading raw dataset from {input_file}...")
    try:
        # Load converted CSV
        df = pd.read_csv(input_file, low_memory=False, encoding='utf-8')
        print(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # 1. Clean and standardize Narrative Text fields
        text_cols = ['NARRATIVE', 'ACCIDENT_DETAILS', 'SHUTDOWN_EXPLAIN']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', 'NULL', ''], np.nan)
                df[col] = df[col].fillna("No narrative provided.")
        
        # 2. Standardize Categorical / Indicator Flags
        flag_cols = ['FATALITY_IND', 'INJURY_IND', 'SHUTDOWN_DUE_ACCIDENT_IND', 'IGNITE_IND']
        for col in flag_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
                # Standardize 'Y'/'YES' to 'YES'
                df[col] = df[col].replace({'Y': 'YES', '1': 'YES', 'TRUE': 'YES'})
                df[col] = df[col].replace({'N': 'NO', '0': 'NO', 'FALSE': 'NO'})
        
        # 3. Numeric conversions for Target Labeling metrics
        # FATAL / INJURE mapping fallback if counts are split across columns
        if 'FATAL' not in df.columns and 'FATALITY_IND' in df.columns:
            df['FATAL'] = np.where(df['FATALITY_IND'] == 'YES', 1, 0)
            
        if 'INJURE' not in df.columns and 'INJURY_IND' in df.columns:
            df['INJURE'] = np.where(df['INJURY_IND'] == 'YES', 1, 0)

        numeric_cols = ['FATAL', 'INJURE', 'UNINTENTIONAL_RELEASE', 'ACCIDENT_PSIG']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                # Fallback zero-initialization if column is missing from subset
                df[col] = 0.0

        # 4. Save cleaned dataframe
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully cleaned data and saved to: {output_file}")
        print(f"Cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    except FileNotFoundError:
        print(f"Error: Target input file '{input_file}' not found. Run convert_text_to_csv.py first.")
    except Exception as e:
        print(f"An unexpected error occurred during data cleaning: {e}")

if __name__ == "__main__":
    clean_data()