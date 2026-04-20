from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

#Load semua model
logreg_model = joblib.load("LogReg_best.pkl")
rfclass_model = joblib.load("RFClass_best.pkl")
rfreg_model = joblib.load("RFReg_best.pkl")

@app.post("/predict")
def predict(data: dict):
    #Convert input JSON ke DataFrame
    input_df = pd.DataFrame([data])

    #Pilih model klasifikasi (default Logistic Regression)
    clf_choice = data.get("clf_choice", "LogReg")

    if clf_choice == "RFClass":
        placement_pred = rfclass_model.predict(input_df)[0]
        model_used = "Random Forest Classifier"
    else:
        placement_pred = logreg_model.predict(input_df)[0]
        model_used = "Logistic Regression"

    result = {
        "model_used": model_used,
        "placement_status": "Placed" if placement_pred == 1 else "Not Placed"
    }

    #Jika Placed, prediksi salary dengan RFReg
    if placement_pred == 1:
        salary_pred = rfreg_model.predict(input_df)[0]
        result["predicted_salary_lpa"] = round(float(salary_pred), 2)
    else:
        result["predicted_salary_lpa"] = None

    return result

# Jalankan dengan: uvicorn main:app --reload

