import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

#Data Ingestion
def load_data(features_path="A.csv", targets_path="A_targets.csv"):
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = pd.merge(features, targets, on="Student_ID")
    df["placement_binary"] = df["placement_status"].apply(lambda x: 1 if x=="Placed" else 0)

    #Drop kolom tidak relevan
    # Alasan:
    # - Student_ID : hanya ID unik, tidak berguna untuk prediksi
    # - tenth_percentage & twelfth_percentage : sangat berkorelasi dengan cgpa (redundansi tinggi)
    # - branch, family_income_level, city_tier, internet_access, part_time_job : korelasi rendah dengan target
    # - internships_completed, certifications_count, hackathons_participated, extracurricular_involvement : kontribusi kecil, bisa jadi noise
    # - study_hours_per_day, backlogs : tidak signifikan secara prediktif
    drop_cols = [
        "Student_ID","family_income_level","branch","tenth_percentage","twelfth_percentage",
        "part_time_job","internet_access","internships_completed","certifications_count",
        "extracurricular_involvement","hackathons_participated","city_tier","backlogs",
        "study_hours_per_day"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

#Preprocessing
def build_preprocessor(X):
    num_features = X.select_dtypes(include=["int64","float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )
    return preprocessor

#Pipeline & Training
def train_and_log(df):
    #Split untuk klasifikasi
    X_class = df.drop(columns=["placement_status","salary_lpa","placement_binary"])
    y_class = df["placement_binary"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    #Split untuk regresi (hanya yang Placed)
    df_reg = df[df["placement_binary"]==1]
    X_reg = df_reg.drop(columns=["placement_status","salary_lpa","placement_binary"])
    y_reg = df_reg["salary_lpa"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    preprocessor_class = build_preprocessor(X_class)
    preprocessor_reg = build_preprocessor(X_reg)

    #Model dictionary
    models_class = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RFClass": RandomForestClassifier(random_state=42),
        "GBClass": GradientBoostingClassifier(random_state=42)
    }
    models_reg = {
        "LinReg": LinearRegression(),
        "RFReg": RandomForestRegressor(random_state=42),
        "GBReg": GradientBoostingRegressor(random_state=42)
    }

    mlflow.set_experiment("Placement_Pipeline")

    best_class_model = None
    best_class_f1 = -1
    best_reg_model = None
    best_reg_r2 = -1

    #Klasifikasi
    for name, model in models_class.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[("preprocessor", preprocessor_class),
                                   ("model", model)])
            pipe.fit(X_train_c, y_train_c)
            y_pred = pipe.predict(X_test_c)
            acc = accuracy_score(y_test_c, y_pred)
            f1 = f1_score(y_test_c, y_pred)

            mlflow.log_param("model", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if f1 > best_class_f1:
                best_class_f1 = f1
                best_class_model = pipe

    #Regresi
    for name, model in models_reg.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[("preprocessor", preprocessor_reg),
                                   ("model", model)])
            pipe.fit(X_train_r, y_train_r)
            y_pred_r = pipe.predict(X_test_r)
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
            r2 = r2_score(y_test_r, y_pred_r)

            mlflow.log_param("model", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if r2 > best_reg_r2:
                best_reg_r2 = r2
                best_reg_model = pipe

    #Simpan model terbaik
    if best_class_model:
        joblib.dump(best_class_model, "BestClass.pkl")
    if best_reg_model:
        joblib.dump(best_reg_model, "BestReg.pkl")

if __name__ == "__main__":
    df = load_data()
    train_and_log(df)

