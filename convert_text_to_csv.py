import pandas as pd
import glob
import os

def inspect_and_merge_phmsa():
    txt_files = glob.glob("*.txt")
    
    if not txt_files:
        print("Error: No .txt files found in directory!")
        return

    print("================ 1. FILE INSPECTION & SCHEMA AUDIT ================")
    
    processed_dfs = []
    
    # Core target mapping dictionary
    # Standard Names: 'NARRATIVE', 'TOTAL_COST', 'FATALITIES', 'INJURIES'
    
    for file in sorted(txt_files):
        print(f"\n File: {file}")
        try:
            # Load raw file
            df_temp = pd.read_csv(file, sep='\t', encoding='latin1', low_memory=False)
            df_temp.columns = df_temp.columns.str.strip().str.replace('"', '').str.replace("'", "")
            
            print(f"   • Total Raw Records: {len(df_temp)}")
            print(f"   • Total Columns: {len(df_temp.columns)}")
            
            # --- Detect Narrative Column ---
            narr_col = None
            for col in ['NARRATIVE', 'NARRATIVE_TEXT', 'NARRATIVE_DESCRIPTION', 'NARR_TEXT', 'DESCRIPTION']:
                if col in df_temp.columns:
                    narr_col = col
                    break
            
            # --- Detect Cost Column ---
            cost_col = None
            for col in ['TOTAL_COST_CURRENT', 'EST_COST_OPER_PAID', 'TOTAL_COST', 'ESTIMATED_COST']:
                if col in df_temp.columns:
                    cost_col = col
                    break

            print(f"   • Detected Narrative Column: '{narr_col}'")
            print(f"   • Detected Primary Cost Column: '{cost_col}'")
            
            if narr_col:
                # Filter valid narrative rows
                valid_narrs = df_temp.dropna(subset=[narr_col])
                sample_snippet = str(valid_narrs[narr_col].iloc[0])[:120].replace('\n', ' ')
                print(f"   • Valid Narratives Count: {len(valid_narrs)}")
                print(f"   • Sample Text: \"{sample_snippet}...\"")
                
                # Standardize columns for this dataframe
                df_standard = df_temp.copy()
                df_standard = df_standard.rename(columns={
                    narr_col: 'NARRATIVE',
                })
                if cost_col:
                    df_standard = df_standard.rename(columns={cost_col: 'EST_COST_OPER_PAID'})
                
                # Tag original source for tracking
                df_standard['SOURCE_FILE'] = file
                
                processed_dfs.append(df_standard)
            else:
                print("    WARNING: Could not identify a narrative column in this file. Skipping.")
                
        except Exception as e:
            print(f"Error reading file: {e}")

    print("\n================ 2. SCHEMA STANDARDIZATION & MERGE ================")
    
    if not processed_dfs:
        print("Error: No valid dataframes to merge.")
        return

    # Merge verified dataframes
    merged_df = pd.concat(processed_dfs, axis=0, ignore_index=True)
    
    # Save output matching downstream scripts
    output_filename = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv'
    merged_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"\n SUCCESS!")
    print(f"Merged {len(processed_dfs)} datasets into: '{output_filename}'")
    print(f"Final Matrix Shape: {merged_df.shape[0]} total incidents across {merged_df.shape[1]} attributes.")
    print("==================================================================")

if __name__ == "__main__":
    inspect_and_merge_phmsa()
