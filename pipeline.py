import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import mlflow
import mlflow.sklearn
import joblib

#Data Ingestion
def load_data(features_path="A.csv", targets_path="A_targets.csv"):
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = pd.merge(features, targets, on="Student_ID")
    df["placement_binary"] = df["placement_status"].apply(lambda x: 1 if x=="Placed" else 0)

    #Drop kolom yang tidak relevan
    drop_cols = [
        "Student_ID", "family_income_level", "branch", "tenth_percentage", "twelfth_percentage",
        "part_time_job", "internet_access", "internships_completed", "certifications_count",
        "extracurricular_involvement", "hackathons_participated", "city_tier", "backlogs",
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

    models = {
        "LogReg": (LogisticRegression(max_iter=1000), preprocessor_class, X_train_c, y_train_c, X_test_c, y_test_c, "accuracy"),
        "RFClass": (RandomForestClassifier(), preprocessor_class, X_train_c, y_train_c, X_test_c, y_test_c, "accuracy"),
        "RFReg": (RandomForestRegressor(), preprocessor_reg, X_train_r, y_train_r, X_test_r, y_test_r, "r2")
    }

    mlflow.set_experiment("Placement_Pipeline")
    for name, (model, preproc, X_train, y_train, X_test, y_test, metric) in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[("preprocessor", preproc),
                                   ("model", model)])
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            mlflow.log_metric(metric, score)

            mlflow.sklearn.log_model(pipe, artifact_path="model")
            joblib.dump(pipe, f"{name}_best.pkl")


if __name__ == "__main__":
    df = load_data()
    train_and_log(df)

