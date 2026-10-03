# Fraud_Detection
That is the best example of imbalance data handling . This is used for detecte the fraud transaction according there transaction behaviour and other factor's . There also define the best way to plot chart of pr_auc . in other word means accuracy
# Retail Fraud Detection 

A Machine Learning project built to detect fraudulent retail transactions. The goal of this project is to identify high-risk transactions using classification models and provide an interactive web interface for predictions.

## Project Overview
Fraud detection is highly imbalanced by nature. In this project, I worked with a dataset of 100k retail transactions (`retail_fraud_detection_100k.csv`) to classify transactions into Low, Medium, and High risk. I built an end-to-end ML pipeline covering data preprocessing, model training, evaluation, and deployment.

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Algorithms:** Random Forest, XGBoost, LightGBM
- **Web Framework:** Streamlit
- **Other:** Joblib (for model saving)

##  Key Highlights & Results
- Built a custom `ColumnTransformer` pipeline to handle missing values and encode categorical variables (OneHotEncoding) and scale numerical data.
- Evaluated models using **PR-AUC (Precision-Recall AUC)** instead of plain accuracy due to class imbalance. 
- Achieved a PR-AUC score of **~0.75** on the test set.
- **Top Features Identified:** `unusual_location_flag`, `unusual_amount_flag`, and `failed_transaction_count_24h`.

##  Project Structure
```text
 Fraud_Detection
  app.py                      # Streamlit frontend application
  deployement_fraud.ipynb     # Final notebook for model training & evaluation
  rough_fraud_detection.ipynb # EDA and initial experiments
  pipeline.pkl                # Saved preprocessing pipeline
  model_file.pkl              # Saved trained model
  retail_fraud_detection_100k.csv # Dataset (Sample)
