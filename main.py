from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

#Load model
class_model = joblib.load("BestClass.pkl")
reg_model = joblib.load("BestReg.pkl")

@app.post("/predict")
def predict(data: dict):
    #Convert input JSON ke DataFrame
    input_df = pd.DataFrame([data])

    #Prediksi klasifikasi
    placement_pred = class_model.predict(input_df)[0]
    result = {
        "placement_status": "Placed" if placement_pred == 1 else "Not Placed"
    }

    #Jika Placed, prediksi salary
    if placement_pred == 1:
        salary_pred = reg_model.predict(input_df)[0]
        result["predicted_salary_lpa"] = round(float(salary_pred), 2)
    else:
        result["predicted_salary_lpa"] = None

    return result