import joblib
import pandas as pd

def load_model():
    return joblib.load("models/xgb_regressor_phase2_corrected.joblib")

def predict_price(model, route, date, time, cab_type):
    df = pd.DataFrame([{
        "route": route,
        "date": str(date),
        "time": str(time),
        "cab_type": cab_type
    }])
    return model.predict(df)[0]
