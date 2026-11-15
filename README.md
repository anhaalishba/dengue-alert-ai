# DengueAlert AI

Dengue outbreak forecasting (XGBoost + LSTM) with a RAG-based health advisory (WHO/NIH guidelines + LLM).

## Setup
    pip install -r requirements.txt

## Data (IMPORTANT)
Download DengAI from DrivenData / Kaggle and put these in `data/`:
- dengue_features_train.csv
- dengue_labels_train.csv

Without them the code uses SYNTHETIC data (only for testing the pipeline). Be honest about this in your viva.
To switch city: `load_data(city="iq")` (default "sj" = San Juan).

## Run (in order)
    python train_xgb.py      # naive vs XGBoost, SHAP, outbreak recall -> outputs/
    python train_lstm.py     # LSTM comparison, prints final table
    streamlit run app.py     # demo

## RAG documents
Put real WHO / NIH dengue guidance as .txt files in `docs/` (paragraphs separated by blank lines).
`docs/dengue_guidelines.txt` is only a starter note (paraphrased general guidance): replace/extend it and cite real sources.

