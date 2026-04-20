import streamlit as st
import requests

st.title("Placement Prediction Client")
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

#Pilihan model klasifikasi
clf_choice = st.sidebar.selectbox("Classification Model", ["LogReg", "RFClass"])

#Buat payload JSON
payload = {
    "gender": gender,
    "degree": degree,
    "cgpa": cgpa,
    "internships": internships,
    "attendance_percentage": attendance,
    "projects_completed": projects,
    "aptitude_skill_rating": aptitude,
    "coding_skill_rating": coding,
    "communication_skill_rating": comm_skill,
    "stress_level": stress,
    "sleep_hours": sleep_hours,
    "clf_choice": clf_choice
}

#Kirim request ke FastAPI
if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()
    st.write("**Model Used:**", result["model_used"])
    st.write("**Placement Status:**", result["placement_status"])
    if result["predicted_salary_lpa"]:
        st.write("**Predicted Salary (LPA):**", result["predicted_salary_lpa"])

