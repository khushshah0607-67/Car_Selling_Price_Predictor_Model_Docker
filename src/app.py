import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "Car Resale Value Predictor API")

model = joblib.load('../models/linear_regression_model.pkl')
scaler = joblib.load('../models/feature_scaler.pkl')
encoder = joblib.load('../models/categorical_encoder.pkl')

class CarFeatures(BaseModel):
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int
    Car_Age: int

@app.post("/predict")
def predict_car_price(data: CarFeatures):
    raw_input = data.model_dump()
    input_df = pd.DataFrame([raw_input])

    input_df['log_present_price'] = np.log1p(input_df['Present_Price'])

    cat_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
    encoded_array = encoder.transform(input_df[cat_cols])
    encoded_columns = encoder.get_feature_names_out(cat_cols)
    df_encoded = pd.DataFrame(encoded_array, columns = encoded_columns, index = input_df.index)

    scale_cols = ['Kms_Driven', 'Car_Age', 'log_present_price']
    input_df_scaled = input_df.copy()
    input_df_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

    X_inference = pd.concat([input_df_scaled, df_encoded], axis=1)

    features_list = ['Kms_Driven',
        'Owner',
        'log_present_price',
        'Car_Age',
        'Fuel_Type_Diesel',
        'Fuel_Type_Petrol',
        'Seller_Type_Individual',
        'Transmission_Manual']
    X_inference = X_inference[features_list]

    log_prediction = model.predict(X_inference)

    final_prediction = np.expm1(float(log_prediction[0]))

    return {
        'status' : 'success',
        'predicted_value_lakhs' : round(final_prediction, 2)
    }