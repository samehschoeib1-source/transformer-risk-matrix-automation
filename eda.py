import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def run_eda():
    # Load engineered dataset to reflect clean features and dual-axis risk targets
    input_file = 'engineered_risk_data.csv'
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    except FileNotFoundError:
        # Fallback to standard raw file if engineered file is missing
        input_file = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv'
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"Fallback to cleaned dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # Set up visual aesthetics
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Exploratory Data Analysis: PHMSA Regulated Gas Gathering Incidents', fontsize=18, fontweight='bold')

    # --- 1. Incidents Over Time (Yearly Breakdown) ---
    year_col = 'IYEAR' if 'IYEAR' in df.columns else 'INSTALLATION_YEAR'
    if year_col in df.columns:
        year_counts = df[year_col].dropna().astype(int).value_counts().sort_index()
        sns.barplot(x=year_counts.index, y=year_counts.values, ax=axes[0, 0], palette="Blues_d")
        axes[0, 0].set_title('Incidents Timeline Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Incidents')
        axes[0, 0].set_xlabel('Year')

    # --- 2. Top 5 Operators by Incident Count ---
    op_col = 'NAME' if 'NAME' in df.columns else 'OPERATOR_NAME'
    if op_col in df.columns:
        top_operators = df[op_col].value_counts().head(5)
        sns.barplot(x=top_operators.values, y=top_operators.index, ax=axes[0, 1], palette="Oranges_r")
        axes[0, 1].set_title('Top 5 Operators with Most Incidents', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Incident Count')

    # --- 3. Y-Axis: Impact Severity Target Distribution ---
    if 'IMPACT_SEVERITY' in df.columns:
        impact_map = {0: '0: Low', 1: '1: Medium', 2: '2: High', 3: '3: Critical'}
        impact_counts = df['IMPACT_SEVERITY'].map(impact_map).value_counts().sort_index()
        sns.barplot(x=impact_counts.index, y=impact_counts.values, ax=axes[1, 0], palette="Purples_r")
        axes[1, 0].set_title('Target 1: Impact Severity Distribution (Y-Axis)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Incident Count')
        axes[1, 0].set_xlabel('Severity Level')

    # --- 4. X-Axis: Escalation Likelihood Target Distribution ---
    if 'ESCALATION_LIKELIHOOD' in df.columns:
        lik_map = {0: '0: Low', 1: '1: Medium', 2: '2: High'}
        lik_counts = df['ESCALATION_LIKELIHOOD'].map(lik_map).value_counts().sort_index()
        sns.barplot(x=lik_counts.index, y=lik_counts.values, ax=axes[1, 1], palette="Reds_r")
        axes[1, 1].set_title('Target 2: Escalation Likelihood Distribution (X-Axis)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Incident Count')
        axes[1, 1].set_xlabel('Likelihood Level')

    # --- 5. Financial Cost Breakdown (Distribution Analysis) ---
    cost_cols = [c for c in ['EST_COST_OPER_PAID', 'EST_COST_PROP_DAMAGE', 'EST_COST_EMERGENCY', 'EST_COST_OTHER'] if c in df.columns]
    if cost_cols:
        cost_summary = df[cost_cols].sum().reset_index()
        cost_summary.columns = ['Cost Category', 'Total USD']
        cost_summary['Cost Category'] = cost_summary['Cost Category'].str.replace('EST_COST_', '')

        sns.barplot(data=cost_summary, x='Cost Category', y='Total USD', ax=axes[2, 0], palette="rocket")
        axes[2, 0].set_title('Total Estimated Financial Costs (USD)', fontsize=12, fontweight='bold')
        axes[2, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # --- 6. Executive Summary Text Card ---
    axes[2, 1].axis('off')
    
    # Calculate values dynamically
    total_incidents = len(df)
    fatal_sum = df['FATAL'].sum() if 'FATAL' in df.columns else 0
    injure_sum = df['INJURE'].sum() if 'INJURE' in df.columns else 0
    prop_damage = df['EST_COST_PROP_DAMAGE'].sum() if 'EST_COST_PROP_DAMAGE' in df.columns else 0
    top_op_str = top_operators.index[0] if op_col in df.columns and len(top_operators) > 0 else "N/A"

    summary_text = (
        f"--- Key Capstone Insights Summary ---\n\n"
        f"• Total Analyzed Incidents: {total_incidents}\n"
        f"• Human Casualties: {int(fatal_sum)} Fatalities, {int(injure_sum)} Injuries\n"
        f"• Total Property Damage: ${prop_damage:,.2f}\n"
        f"• Leading Operator: {top_op_str}\n"
        f"• Target Balance (Impact): Class 0 ({impact_counts.values[0] if 'IMPACT_SEVERITY' in df.columns else 0} recs)\n"
        f"• Target Balance (Likelihood): Class 0 ({lik_counts.values[0] if 'ESCALATION_LIKELIHOOD' in df.columns else 0} recs)"
    )
    axes[2, 1].text(0.05, 0.35, summary_text, fontsize=13, weight='normal',
                    bbox=dict(facecolor='lightgray', alpha=0.3, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save chart image to disk for README/Paper insertion
    plt.savefig('eda_dashboard.png', dpi=300)
    print(" Dashboard saved successfully as 'eda_dashboard.png'.")
    plt.show()

    # --- Print Summary Descriptive Stats to Console ---
    if cost_cols:
        print("\n=== COST STATISTICAL SUMMARY ===")
        print(df[cost_cols].describe().round(2))

if __name__ == "__main__":
    run_eda()
