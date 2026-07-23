# Transformer-Based Risk Matrix Automation in Oil & Gas

## Project Overview
This repository contains the complete open-access replication package, datasets, and codebase for the graduate data analytics capstone project: **Transformer-Based Risk Matrix Automation in Oil & Gas**.

The objective of this research is to automate the extraction and mapping of operational risks from unstructured federal incident narratives (U.S. DOT/PHMSA data) onto an active dual-axis risk matrix. This is accomplished by leveraging multi-task Deep Learning and Transformer-based NLP architectures (DistilBERT) alongside classical machine learning benchmarks.

---

## Repository Structure
```plaintext
├── accident_hazardous_liquid_jan2010_present.txt                          <- Raw PHMSA Dataset (Liquid Pipelines & Gathering)
├── incident_gas_transmission_gathering_2002_dec2009.txt                   <- Raw PHMSA Dataset (Historical Transmission & Gathering)
├── incident_gas_transmission_gathering_jan2010_present.txt                <- Raw PHMSA Dataset (Gas Transmission & Gathering)
├── incident_type_r_reporting_regulated_gas_gathering_may2022_present.txt  <- Raw PHMSA Dataset (Type R Gathering)
├── incident_type_r_reporting_regulated_gas_gathering_may2022_present.csv  <- Unified Standardized CSV Matrix (9,117 Rows)
├── convert_text_to_csv.py                                                 <- Batch Data Extraction, Schema Alignment & Merge Script
├── clean_data.py                                                          <- Structural Filtering, Text Normalization & Preprocessing
├── engineer_labels.py                                                     <- Dual-Axis Risk Target Label Engineering (Impact & Likelihood)
├── eda.py                                                                 <- Distribution Inspection, Statistical Audits & Dashboard
├── baseline_model.py                                                      <- TF-IDF & Oversampled Random Forest Baseline ML Model
├── transformer_model.py                                                   <- Multi-Task DistilBERT / BERT Fine-Tuning Neural Network
└── README.md                                                              <- Repository Documentation Landing Page
