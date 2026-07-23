import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats

def run_evaluation_comparison():
    print("========================================================================================")
    print("                     CAPSTONE MODEL EVALUATION & COMPARISON REPORT                      ")
    print("========================================================================================\n")

    baseline_path = 'results/baseline_preds.npz'
    transformer_path = 'results/transformer_preds.npz'

    # Check if prediction files exist
    if not os.path.exists(baseline_path) or not os.path.exists(transformer_path):
        print("Error: Missing prediction files in './results/'.")
        print("Please ensure both 'baseline_model.py' and 'transformer_model.py' have been executed.")
        return

    # Load prediction arrays
    base_data = np.load(baseline_path)
    trans_data = np.load(transformer_path)

    # 1. Impact Severity Metrics Extraction
    y_true_imp = base_data['y_true_impact']
    y_pred_imp_base = base_data['y_pred_impact']
    y_pred_imp_trans = trans_data['y_pred_impact']

    acc_imp_base = accuracy_score(y_true_imp, y_pred_imp_base)
    acc_imp_trans = accuracy_score(y_true_imp, y_pred_imp_trans)

    report_imp_base = classification_report(y_true_imp, y_pred_imp_base, output_dict=True, zero_division=0)
    report_imp_trans = classification_report(y_true_imp, y_pred_imp_trans, output_dict=True, zero_division=0)

    # Class 3 (Critical Impact) Specific Metrics
    c3_rec_base = report_imp_base.get('3', {}).get('recall', 0.0)
    c3_rec_trans = report_imp_trans.get('3', {}).get('recall', 0.0)
    c3_f1_base = report_imp_base.get('3', {}).get('f1-score', 0.0)
    c3_f1_trans = report_imp_trans.get('3', {}).get('f1-score', 0.0)

    # 2. Escalation Likelihood Metrics Extraction
    y_true_esc = base_data['y_true_escalation']
    y_pred_esc_base = base_data['y_pred_escalation']
    y_pred_esc_trans = trans_data['y_pred_escalation']

    acc_esc_base = accuracy_score(y_true_esc, y_pred_esc_base)
    acc_esc_trans = accuracy_score(y_true_esc, y_pred_esc_trans)

    report_esc_base = classification_report(y_true_esc, y_pred_esc_base, output_dict=True, zero_division=0)
    report_esc_trans = classification_report(y_true_esc, y_pred_esc_trans, output_dict=True, zero_division=0)

    # Class 2 (High Escalation) Specific Metrics
    c2_prec_base = report_esc_base.get('2', {}).get('precision', 0.0)
    c2_prec_trans = report_esc_trans.get('2', {}).get('precision', 0.0)
    c2_rec_base = report_esc_base.get('2', {}).get('recall', 0.0)
    c2_rec_trans = report_esc_trans.get('2', {}).get('recall', 0.0)
    c2_f1_base = report_esc_base.get('2', {}).get('f1-score', 0.0)
    c2_f1_trans = report_esc_trans.get('2', {}).get('f1-score', 0.0)

    # Print Formatted ASCII Benchmarking Table
    print(f"{'Metric / Classification Target':<40} | {'Baseline (RF)':<15} | {'Transformer (DistilBERT)':<25} | {'Δ Diff':<10}")
    print("-" * 98)
    
    print(f"{'Impact Severity Accuracy':<40} | {acc_imp_base*100:>13.2f}% | {acc_imp_trans*100:>23.2f}% | {(acc_imp_trans - acc_imp_base)*100:>+8.2f}%")
    print(f"{'  - Class 3 (Critical) Recall':<40} | {c3_rec_base*100:>13.2f}% | {c3_rec_trans*100:>23.2f}% | {(c3_rec_trans - c3_rec_base)*100:>+8.2f}%")
    print(f"{'  - Class 3 (Critical) F1-Score':<40} | {c3_f1_base:>14.2f}  | {c3_f1_trans:>24.2f}  | {c3_f1_trans - c3_f1_base:>+9.2f}")
    
    print("-" * 98)
    print(f"{'Escalation Likelihood Accuracy':<40} | {acc_esc_base*100:>13.2f}% | {acc_esc_trans*100:>23.2f}% | {(acc_esc_trans - acc_esc_base)*100:>+8.2f}%")
    print(f"{'  - Class 2 (High) Precision':<40} | {c2_prec_base*100:>13.2f}% | {c2_prec_trans*100:>23.2f}% | {(c2_prec_trans - c2_prec_base)*100:>+8.2f}%")
    print(f"{'  - Class 2 (High) Recall':<40} | {c2_rec_base*100:>13.2f}% | {c2_rec_trans*100:>23.2f}% | {(c2_rec_trans - c2_rec_base)*100:>+8.2f}%")
    print(f"{'  - Class 2 (High) F1-Score':<40} | {c2_f1_base:>14.2f}  | {c2_f1_trans:>24.2f}  | {c2_f1_trans - c2_f1_base:>+9.2f}")
    print("=" * 98)

    # 3. Two-Sample Proportion Z-Test Calculation (RQ4 Validation)
    # Total correct matrix placements (Both Impact AND Escalation predicted correctly)
    correct_base = np.sum((y_pred_imp_base == y_true_imp) & (y_pred_esc_base == y_true_esc))
    correct_trans = np.sum((y_pred_imp_trans == y_true_imp) & (y_pred_esc_trans == y_true_esc))
    
    n_samples = len(y_true_imp)
    p1 = correct_trans / n_samples
    p2 = correct_base / n_samples
    
    # Pooled proportion
    p_pool = (correct_trans + correct_base) / (2 * n_samples)
    se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_samples))
    
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print("\n----------------------------------------------------------------------------------------")
    print("                 RQ4 HYPOTHESIS TEST RESULTS (Two-Sample Proportion Z-Test)              ")
    print("----------------------------------------------------------------------------------------")
    print(f"Total Test Samples (N):                         {n_samples}")
    print(f"Baseline Matrix Joint Predictions Correct:      {correct_base} ({p2*100:.2f}%)")
    print(f"Transformer Matrix Joint Predictions Correct:   {correct_trans} ({p1*100:.2f}%)")
    print(f"Calculated Z-Statistic:                        {z_stat:.4f}")
    print(f"Calculated P-Value:                            {p_value:.4e}")
    
    if p_value < 0.05:
        print("Conclusion:                                    Statistically Significant (Reject H0)")
        print("                                               The Transformer model demonstrates statistically")
        print("                                               superior overall risk matrix assignment (p < 0.05).")
    else:
        print("Conclusion:                                    Fail to Reject H0 (Not Statistically Significant)")
    print("========================================================================================\n")

if __name__ == "__main__":
    run_evaluation_comparison()