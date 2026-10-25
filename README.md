# ML-Based Network Intrusion Detection and Analysis System

A beginner-friendly Python project that trains a machine-learning classifier on labelled network-flow data and provides a Streamlit dashboard for exploring predictions.

## Project features

- Load a labelled CSV network-flow dataset.
- Preprocess numeric and categorical columns.
- Train and compare a Decision Tree and Random Forest classifier.
- Report accuracy, precision, recall, F1-score, and a confusion matrix.
- Save the trained model and preprocessing pipeline.
- Upload a compatible CSV in the Streamlit app and download predictions.
- View feature importance when supported by the trained model.

## Technology stack

- **Language:** Python 3.10+
- **Machine learning:** scikit-learn (Decision Tree, Random Forest, preprocessing, evaluation)
- **Data processing:** pandas, NumPy
- **Dashboard / frontend:** Streamlit
- **Backend / model logic:** Python functions in `src/`
- **Charts:** Matplotlib and Seaborn
- **Model persistence:** joblib
- **API:** No external API is required. Streamlit runs the Python application locally.

## Dataset

Recommended public dataset: **CIC-IDS2017**, provided by the Canadian Institute for Cybersecurity.

Official page: https://www.unb.ca/cic/datasets/ids-2017.html

Download a labelled CSV flow file from the official page. The project expects one row per flow and a target/label column such as `Label`. Dataset file names and column names can vary. Keep the downloaded CSV in the `data/` folder and pass its path to the training script.

> Use public datasets for this coursework. This project does not sniff live network traffic or scan any network.

## Setup (Windows PowerShell)

1. Install Python 3.10 or later.
2. Extract the ZIP file and open PowerShell in the extracted folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install libraries:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Download a labelled CSV from the dataset link above and save it as `data/network_traffic.csv`.
6. Train the model:

   ```powershell
   python src/train_model.py --data data/network_traffic.csv --target Label
   ```

   If the label column has a different name, replace `Label` with the exact column name from your CSV. You can inspect column names with `python -c "import pandas as pd; print(pd.read_csv('data/network_traffic.csv', nrows=2).columns.tolist())"`.

7. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

8. In the dashboard, upload a compatible CSV. The uploaded file should have the same input feature columns used during training; the label column is optional for prediction.

## Expected outputs

- `models/intrusion_model.joblib` — trained pipeline
- `reports/metrics.json` — evaluation metrics
- `reports/confusion_matrix.png` — confusion matrix plot
- `reports/feature_importance.png` — feature importance plot when available

## Limitations

- This is a learning/demo project, not a production IDS.
- Prediction quality depends on dataset quality and the selected features.
- A model trained on one dataset may not generalize to real networks.
- Labels and class balance should be checked before making claims about detection quality.
