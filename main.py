
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import root_mean_squared_error


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attri, cat_attri):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attri),
        ('cat', cat_pipeline, cat_attri)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("housing.csv")
    data['income_cat'] = pd.cut(
        data['median_income'],
        bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]   

    strat_test_set.to_csv("input.csv", index=False)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy() 

    num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)
 
input_data = pd.read_csv("input.csv")

input_labels = input_data["median_house_value"].copy()

transformed_input = pipeline.transform(input_data)
forest_preds = model.predict(transformed_input)
input_data["median_house_value"] = forest_preds

forest_rmse = root_mean_squared_error(input_labels, forest_preds)
print("Random Forest RMSE:", forest_rmse)

input_data.to_csv("output.csv", index=False)
print("Inference complete. Results saved to output.csv")
Random Forest RMSE: 47119.62863546612
Inference complete. Results saved to output.csv
 
