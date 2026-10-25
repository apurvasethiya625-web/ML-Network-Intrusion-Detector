# Technology Stack

- **Language:** Python 3.10+
- **Notebook:** Jupyter Notebook (`.ipynb`)
- **Data handling:** pandas, NumPy
- **ML algorithms:** Decision Tree and Random Forest (`scikit-learn`)
- **Preprocessing:** SimpleImputer, OneHotEncoder, ColumnTransformer, Pipeline
- **Evaluation:** accuracy, macro precision, macro recall, macro F1, classification report, confusion matrix
- **Visualization:** Matplotlib, Seaborn
- **Frontend/dashboard:** Streamlit (web UI running locally)
- **Backend/model logic:** Python scripts and scikit-learn pipeline
- **Saved model:** joblib
- **External APIs:** none required
- **Dataset:** CIC-IDS2017 (official page: https://www.unb.ca/cic/datasets/ids-2017.html)

## Data flow
CSV upload/download → pandas → preprocessing → trained ML pipeline → predictions and evaluation → Streamlit display.
