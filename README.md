# House Price Predictor

This project trains a machine learning model to predict housing prices based on the California Housing dataset. It uses a preprocessing and modeling pipeline built with scikit-learn and saves the trained model for future inference.

The system automatically:
- Loads the dataset
- Splits it using stratified sampling based on income categories
- Preprocesses numerical and categorical features
- Trains a Random Forest model
- Saves both the preprocessing pipeline and trained model
- Runs predictions on unseen test data
- Generates a CSV file with predicted values

---

## 📁 Files in the Project

| File | Purpose |
|------|---------|
| `housing.csv` | Raw dataset used for training |
| `main.py` | Main script to train model and generate predictions |
| `model.pkl` | Saved trained Random Forest model |
| `pipeline.pkl` | Saved preprocessing pipeline |
| `input.csv` | Test data used for prediction (generated automatically on first run) |
| `output.csv` | Final prediction output with model predictions |

---

## ⚙️ How It Works

The script first checks if a saved model exists.

- If the model is not found, it trains a new model, generates an `input.csv` file, and saves the model and preprocessing pipeline.
- If the model already exists, the script loads it and performs inference using the saved pipeline.

### Workflow Steps

1. Load dataset
2. Create income categories for stratified sampling
3. Apply preprocessing:
   - Numerical features: imputation + scaling
   - Categorical features: imputation + one-hot encoding
4. Train a Random Forest Regressor
5. Evaluate performance using RMSE
6. Save predictions into `output.csv`

The script prints the model’s RMSE score after prediction.

---

## ▶️ Running the Project

### 1. Install Dependencies

Run the following command:

```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Execute the Script
```bash
python main.py
```
After running, you should see an output similar to:
```bash
Model trained and saved.
Random Forest RMSE: 47119.62863546612
Inference complete. Results saved to output.csv
```
