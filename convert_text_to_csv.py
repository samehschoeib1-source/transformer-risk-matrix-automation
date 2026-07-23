import pandas as pd

def convert_dataset():
    txt_file_path = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.txt'
    csv_file_path = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv'
    
    print(f"Initiating extraction from raw file: {txt_file_path}...")
    
    try:
        # Using cp1252 to gracefully handle embedded special characters found in report text fields
        df = pd.read_csv(txt_file_path, sep='\t', encoding='cp1252')
        
        # Save to standard CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        
        print(f"Success! The text file has been converted and saved to: {csv_file_path}")
        print(f"Extracted shape matrix: {df.shape[0]} records across {df.shape[1]} raw attributes.")
        
    except FileNotFoundError:
        print(f"Error: The target file '{txt_file_path}' was not found in the current working directory.")
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")

if __name__ == "__main__":
    convert_dataset()
