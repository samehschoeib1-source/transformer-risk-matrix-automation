# Transformer-Based Risk Matrix Automation in Oil & Gas

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview
This repository contains the complete open-access replication package, datasets, and codebase for the graduate data analytics capstone project: **Transformer-Based Risk Matrix Automation in Oil & Gas**.

The objective of this research is to automate the extraction and mapping of operational risks from unstructured federal incident narratives (U.S. DOT/PHMSA data) onto an active dual-axis risk matrix. This is accomplished by leveraging multi-task Deep Learning and Transformer-based NLP architectures (DistilBERT) alongside classical machine learning benchmarks.

---

## Repository Structure
```plaintext
├── accident_hazardous_liquid_jan2010_present.txt            <- Raw PHMSA Dataset (Liquid Pipelines & Gathering)
├── incident_gas_transmission_gathering_2002_dec2009.txt     <- Raw PHMSA Dataset (Historical Transmission & Gathering)
├── incident_gas_transmission_gathering_jan2010_present.txt  <- Raw PHMSA Dataset (Gas Transmission & Gathering)
├── incident_type_r_reporting_regulated_gas_gathering_may2022_present.txt  <- Raw PHMSA Dataset (Type R Gathering)
├── incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv  <- Unified Standardized CSV Matrix (9,117 Rows)
├── convert_text_to_csv.py                                   <- Batch Data Extraction, Schema Alignment & Merge Script
├── clean_data.py                                            <- Structural Filtering, Text Normalization & Preprocessing
├── engineer_labels.py                                       <- Dual-Axis Risk Target Label Engineering (Impact & Likelihood)
├── eda.py                                                   <- Distribution Inspection, Statistical Audits & Dashboard
├── plots/                                                   <- Saved EDA Visualizations & Charts
│   ├── class_distributions.png                              <- Impact & Likelihood Label Distribution Bar Charts
│   └── risk_matrix_heatmap.png                              <- Co-occurrence Heatmap of Risk Class Combinations
├── baseline_model.py                                        <- TF-IDF & Oversampled Random Forest Baseline ML Model
├── transformer_model.py                                     <- Multi-Task DistilBERT / BERT Fine-Tuning Neural Network
├── evaluate_comparison.py                                   <- Comparative Benchmarking, ASCII Table & Z-Test (RQ4)
├── results/                                                 <- Exported Prediction Artifacts Directory
│   ├── baseline_preds.npz                                   <- Saved Test Predictions & Labels (Random Forest)
│   └── transformer_preds.npz                                <- Saved Test Predictions & Labels (DistilBERT)
└── README.md                                                <- Repository Documentation Landing Page
```
---

## License & Data Source
* **License:** MIT License
* **Data Portal:** Source records compiled from the [U.S. DOT PHMSA Incident Data Portal](https://www.phmsa.dot.gov/data-and-statistics/pipeline/distribution-transmission-gathering-lng-and-liquid-accident-and-incident-data).
