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
    "\n",
    "def convert_dataset():\n",
    "    txt_file_path = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.txt'\n",
    "    csv_file_path = 'incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv'\n",
    "    \n",
    "    print(f\"Initiating extraction from raw file: {txt_file_path}...\")\n",
    "    \n",
    "    try:\n",
    "        # Using cp1252 to gracefully handle embedded special characters found in report text fields\n",
    "        df = pd.read_csv(txt_file_path, sep='\\t', encoding='cp1252')\n",
    "        \n",
    "        # Fixed: Added missing comma for index=False parameter\n",
    "        df.to_csv(csv_file_path, index=False, encoding='utf-8')\n",
    "        \n",
    "        print(f\"Success! The text file has been converted and saved to: {csv_file_path}\")\n",
    "        print(f\"Extracted shape matrix: {df.shape[0]} records across {df.shape[1]} raw attributes.\")\n",
    "        \n",
    "    except FileNotFoundError:\n",
    "        print(f\"Error: The target file '{txt_file_path}' was not found in the current working directory.\")\n",
    "    except Exception as e:\n",
    "        print(f\"An unexpected error occurred during parsing: {e}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    convert_dataset()"
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
