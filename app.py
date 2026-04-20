import streamlit as st
import pandas as pd
import joblib

#Load Models
logreg_model = joblib.load("LogReg_best.pkl")
rfclass_model = joblib.load("RFClass_best.pkl")
rfreg_model = joblib.load("RFReg_best.pkl")

#UI/UX Design
st.title("Placement Prediction App")
st.sidebar.header("Input Features")

#Input
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
degree = st.sidebar.selectbox("Degree", ["B.Tech", "M.Tech", "MBA"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.5)
internships = st.sidebar.number_input("Internships", min_value=0, max_value=10, value=1)
attendance = st.sidebar.slider("Attendance Percentage", 0, 100, 85)
projects = st.sidebar.number_input("Projects Completed", min_value=0, max_value=20, value=2)
aptitude = st.sidebar.slider("Aptitude Skill Rating", 0, 10, 5)
coding = st.sidebar.slider("Coding Skill Rating", 0, 10, 6)
comm_skill = st.sidebar.slider("Communication Skill Rating", 0, 10, 7)
stress = st.sidebar.slider("Stress Level", 0, 10, 5)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 24, 7)

#Buat DataFrame input
input_data = pd.DataFrame({
    "gender":[gender],
    "degree":[degree],
    "cgpa":[cgpa],
    "internships":[internships],
    "attendance_percentage":[attendance],
    "projects_completed":[projects],
    "aptitude_skill_rating":[aptitude],
    "coding_skill_rating":[coding],
    "communication_skill_rating":[comm_skill],
    "stress_level":[stress],
    "sleep_hours":[sleep_hours]
})

#Prediction Logic
st.subheader("Prediction Results")
clf_choice = st.selectbox("Choose Classification Model", ["Logistic Regression", "Random Forest Classifier"])

if clf_choice == "Logistic Regression":
    placement_pred = logreg_model.predict(input_data)[0]
else:
    placement_pred = rfclass_model.predict(input_data)[0]

placement_label = "Placed" if placement_pred == 1 else "Not Placed"
st.write(f"**Placement Status Prediction ({clf_choice}):** {placement_label}")

if placement_pred == 1:
    salary_pred = rfreg_model.predict(input_data)[0]
    st.write(f"**Predicted Salary (LPA):** {salary_pred:.2f}")
else:
    st.write("**Predicted Salary (LPA):** N/A (Not Placed)")

#Visualization
if placement_pred == 1:
    st.bar_chart(pd.DataFrame({"Predicted Salary":[salary_pred]}))


