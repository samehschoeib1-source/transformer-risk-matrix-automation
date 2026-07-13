{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1d1a8fe-43ee-4756-b60a-acfec2e23127",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "def engineer_risk_targets():\n",
    "    input_file = 'cleaned_incident_data.csv'\n",
    "    output_file = 'engineered_risk_data.csv'\n",
    "    \n",
    "    print(f\"Loading cleaned dataset: {input_file}...\")\n",
    "    try:\n",
    "        df = pd.read_csv(input_file, encoding='utf-8')\n",
    "        \n",
    "        # --- 1. Target Axis 1: Impact Severity ---\n",
    "        print(\"Engineering Impact Severity categories...\")\n",
    "        conditions_severity = [\n",
    "            (df['FATAL'] > 0) | (df['INJURE'] > 0), # Critical\n",
    "            (df['SHUTDOWN_DUE_ACCIDENT_IND'] == 'YES'), # High\n",
    "            (df['UNINTENTIONAL_RELEASE'] > df['UNINTENTIONAL_RELEASE'].median()) # Medium\n",
    "        ]\n",
    "        choices_severity = [3, 2, 1]\n",
    "        # Default fallback is 0 (Low)\n",
    "        df['IMPACT_SEVERITY'] = np.select(conditions_severity, choices_severity, default=0)\n",
    "        \n",
    "        # --- 2. Target Axis 2: Escalation Likelihood ---\n",
    "        print(\"Engineering Escalation Likelihood categories...\")\n",
    "        # Higher risk if it ignited or happened on a highly pressurized line segment\n",
    "        median_psig = df['ACCIDENT_PSIG'].median()\n",
    "        conditions_likelihood = [\n",
    "            (df['IGNITE_IND'] == 'YES'), # High Risk\n",
    "            (df['ACCIDENT_PSIG'] > median_psig) # Medium Risk\n",
    "        ]\n",
    "        choices_likelihood = [2, 1]\n",
    "        # Default fallback is 0 (Low Risk)\n",
    "        df['ESCALATION_LIKELIHOOD'] = np.select(conditions_likelihood, choices_likelihood, default=0)\n",
    "        \n",
    "        # Print class distribution summaries to verify no empty slices\n",
    "        print(\"\\n=== Target Matrix Distributions ===\")\n",
    "        print(\"Impact Severity (0=Low, 3=Critical):\\n\", df['IMPACT_SEVERITY'].value_counts().sort_index())\n",
    "        print(\"Escalation Likelihood (0=Low, 2=High):\\n\", df['ESCALATION_LIKELIHOOD'].value_counts().sort_index())\n",
    "        \n",
    "        # Save the dataset with its brand new target labels\n",
    "        df.to_csv(output_file, index=False, encoding='utf-8')\n",
    "        print(f\"\\nSaved newly labeled dataset as: {output_file}\")\n",
    "        \n",
    "    except FileNotFoundError:\n",
    "        print(f\"Error: Required file '{input_file}' not found. Run clean_data.py first.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    engineer_risk_targets()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
