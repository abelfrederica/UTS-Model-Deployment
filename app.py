import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

#Load Models
class_model = joblib.load("BestClass.pkl")
reg_model = joblib.load("BestReg.pkl")

#Page setup
st.set_page_config(page_title="Placement Prediction App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Placement Prediction App</h1>", unsafe_allow_html=True)
st.sidebar.header("Input Features")

#Input Features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.5)
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
    "cgpa":[cgpa],
    "attendance_percentage":[attendance],
    "projects_completed":[projects],
    "aptitude_skill_rating":[aptitude],
    "coding_skill_rating":[coding],
    "communication_skill_rating":[comm_skill],
    "stress_level":[stress],
    "sleep_hours":[sleep_hours]
})

#Prediction Logic
placement_pred = class_model.predict(input_data)[0]
placement_label = "Placed" if placement_pred == 1 else "Not Placed"

#Layout hasil
col1, col2 = st.columns(2)
with col1:
    st.metric("Placement Status", placement_label)
with col2:
    if placement_pred == 1:
        salary_pred = reg_model.predict(input_data)[0]
        st.metric("Predicted Salary (LPA)", f"{salary_pred:.2f}")
    else:
        st.metric("Predicted Salary (LPA)", "N/A")

#Visualization
st.markdown("---")
st.markdown("### Visualizations")

if placement_pred == 1:
    # Salary bar chart
    salary_df = pd.DataFrame({"Predicted Salary":[salary_pred]})
    st.bar_chart(salary_df)

#Skill comparison chart
skills = {
    "Aptitude": aptitude,
    "Coding": coding,
    "Communication": comm_skill,
    "Stress": stress,
    "Sleep Hours": sleep_hours
}
skills_df = pd.DataFrame(list(skills.items()), columns=["Skill","Score"])
fig = px.bar(skills_df, x="Skill", y="Score", title="Skill Comparison", color="Score", text="Score")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)
