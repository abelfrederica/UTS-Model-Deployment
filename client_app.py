import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Placement Prediction Client", layout="wide")
st.markdown("<h2 style='text-align: center; color: #2C3E50;'>Placement Prediction Client</h2>", unsafe_allow_html=True)
st.sidebar.header("Input Features")

#Input
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.5)
attendance = st.sidebar.slider("Attendance Percentage", 0, 100, 85)
projects = st.sidebar.number_input("Projects Completed", min_value=0, max_value=20, value=2)
aptitude = st.sidebar.slider("Aptitude Skill Rating", 0, 10, 5)
coding = st.sidebar.slider("Coding Skill Rating", 0, 10, 6)
comm_skill = st.sidebar.slider("Communication Skill Rating", 0, 10, 7)
stress = st.sidebar.slider("Stress Level", 0, 10, 5)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 24, 7)

payload = {
    "gender": gender,
    "cgpa": cgpa,
    "attendance_percentage": attendance,
    "projects_completed": projects,
    "aptitude_skill_rating": aptitude,
    "coding_skill_rating": coding,
    "communication_skill_rating": comm_skill,
    "stress_level": stress,
    "sleep_hours": sleep_hours
}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()

    #Metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Placement Status", result["placement_status"])
    with col2:
        if result["predicted_salary_lpa"]:
            st.metric("Predicted Salary (LPA)", f"{result['predicted_salary_lpa']:.2f}")
        else:
            st.metric("Predicted Salary (LPA)", "N/A")

    #Visualizations
    st.markdown("---")
    st.subheader("Visualizations")

    if result["predicted_salary_lpa"]:
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(pd.DataFrame({"Predicted Salary":[result["predicted_salary_lpa"]]}))
        with col2:
            skills = {
                "Aptitude": aptitude,
                "Coding": coding,
                "Communication": comm_skill,
                "Stress": stress,
                "Sleep Hours": sleep_hours
            }
            st.bar_chart(pd.DataFrame.from_dict(skills, orient="index", columns=["Score"]))
