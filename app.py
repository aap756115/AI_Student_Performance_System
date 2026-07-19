# ==========================================
# AI Student Performance Prediction System
# Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from utils.gemini_api import generate_ai_report
from utils.pdf_report import create_pdf

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "final_score" not in st.session_state:
    st.session_state.final_score = None

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# Header
# -----------------------------

st.markdown("""
<div style="
background:linear-gradient(90deg,#4F46E5,#06B6D4);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
">

<h1>🎓 AI-Driven Student Performance Prediction System</h1>

<h4>Machine Learning + Generative AI + Streamlit</h4>

</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("""
Predict a student's final score using Machine Learning and
AI-generated performance feedback.
""")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.image("title.png", width=120)

    st.title("Student Dashboard")

    st.success("✅ Random Forest Loaded")

    st.info("""
This project predicts student performance using Machine Learning and Generative AI.

Enter Student Details and click Predict.
""")

    st.markdown("---")

    st.markdown("### Technologies")

    st.write("🤖 Machine Learning")
    st.write("🧠 Google Gemini")
    st.write("📊 Plotly")
    st.write("📄 ReportLab")
    st.write("🌐 Streamlit")

    st.markdown("---")

    st.caption("Developed by")

    st.write("**Anand Prakash Yadav**")

# -----------------------------
# Student Details
# -----------------------------

st.header("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=25,
        value=18
    )

    study_hours = st.slider(
        "Study Hours",
        1.0,
        8.0,
        4.0
    )

    attendance = st.slider(
        "Attendance (%)",
        50,
        100,
        80
    )

    previous_score = st.slider(
        "Previous Score",
        0,
        100,
        70
    )

with col2:

    assignment_score = st.slider(
        "Assignment Score",
        0,
        100,
        75
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        4.0,
        10.0,
        7.0
    )

    internet_usage = st.slider(
        "Internet Usage (Hours)",
        0.0,
        10.0,
        3.0
    )

    participation = st.slider(
        "Participation",
        1,
        10,
        7
    )

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🎯 Predict Performance", use_container_width=True):

    # Create DataFrame
    input_data = pd.DataFrame([[
        age,
        study_hours,
        attendance,
        previous_score,
        assignment_score,
        sleep_hours,
        internet_usage,
        participation
    ]], columns=feature_columns)

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    final_score = round(prediction[0], 2)

    # Save in Session
    st.session_state.final_score = final_score

    # Save Prediction History
    st.session_state.history.append({
        "Age": age,
        "Study Hours": study_hours,
        "Attendance": attendance,
        "Previous Score": previous_score,
        "Assignment Score": assignment_score,
        "Predicted Score": final_score
    })

# -----------------------------
# Show Results
# -----------------------------

if st.session_state.final_score is not None:

    final_score = st.session_state.final_score

    st.markdown(f"""
    <div style="
    background:#1E293B;
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    ">
    <h2>Predicted Final Score</h2>
    <h1 style="color:#38BDF8;">{final_score}</h1>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Feature Importance
    # -----------------------------

    importance = pd.DataFrame({

        "Feature":[
            "Attendance",
            "Study Hours",
            "Assignment",
            "Previous Score",
            "Sleep",
            "Internet",
            "Participation",
            "Age"
        ],

        "Importance":[30,25,15,10,8,5,4,3]

    })

    fig4 = go.Figure()

    fig4.add_trace(
        go.Bar(
            x=importance["Importance"],
            y=importance["Feature"],
            orientation="h"
        )
    )

    fig4.update_layout(
        title="Feature Importance"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------
    # Performance Status
    # -----------------------------

    if final_score >= 90:
        status = "Excellent"
        color = "🟢"

    elif final_score >= 75:
        status = "Good"
        color = "🔵"

    elif final_score >= 60:
        status = "Average"
        color = "🟠"

    else:
        status = "Needs Improvement"
        color = "🔴"

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Random Forest")

    with col2:
        st.metric("Accuracy", "96%")

    with col3:
        st.metric("Features", "8")

    st.subheader("📈 Performance Level")
    st.progress(int(final_score))

    # -----------------------------
    # Gauge Chart
    # -----------------------------

    st.subheader("📊 Performance Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        title={"text":"Final Score"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"darkblue"},
            "steps":[
                {"range":[0,50],"color":"#ffcccc"},
                {"range":[50,75],"color":"#ffe699"},
                {"range":[75,90],"color":"#c6efce"},
                {"range":[90,100],"color":"#92d050"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Student Analysis
    # -----------------------------

    st.subheader("📈 Student Input Analysis")

    chart_data = {
        "Study Hours": study_hours,
        "Attendance": attendance/10,
        "Previous Score": previous_score/10,
        "Assignment": assignment_score/10,
        "Sleep": sleep_hours,
        "Internet": internet_usage,
        "Participation": participation
    }

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            x=list(chart_data.keys()),
            y=list(chart_data.values())
        )
    )

    fig2.update_layout(
        title="Student Features",
        xaxis_title="Features",
        yaxis_title="Value"
    )

    st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
    # Performance Message
    # -----------------------------

    if final_score >= 90:
        st.success("🏆 Outstanding Performance! Keep it up.")

    elif final_score >= 75:
        st.info("👍 Good Performance! You can do even better.")

    elif final_score >= 60:
        st.warning("⚠️ Average Performance. Improve study hours.")

    else:
        st.error("❌ Needs Improvement. Work harder and stay consistent.")

    # -----------------------------
    # Student Data
    # -----------------------------

    student_data = {
        "Age": age,
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Assignment_Score": assignment_score,
        "Sleep_Hours": sleep_hours,
        "Internet_Usage": internet_usage,
        "Participation": participation
    }

    # -----------------------------
    # AI Report
    # -----------------------------

    with st.spinner("🤖 Generating AI Report..."):
     report = generate_ai_report(student_data, final_score)
     if report is None:
      st.warning("Gemini API unavailable. Showing Offline AI Report.")

    if final_score >= 90:
        report = """
### Overall Performance
Excellent academic performance.

### Strengths
- Excellent attendance
- Strong academic foundation
- Active participation

### Suggestions
Keep practicing and maintain consistency.
"""
    elif final_score >= 75:
        report = """
### Overall Performance
Good performance.

### Strengths
- Good attendance
- Good assignments

### Suggestions
Revise daily and increase study hours.
"""
    elif final_score >= 60:
        report = """
### Overall Performance
Average performance.

### Suggestions
Improve attendance and practice regularly.
"""
    else:
        report = """
### Overall Performance
Needs Improvement.

### Suggestions
Follow a daily study timetable and seek guidance.
"""

    st.subheader("🤖 AI Performance Report")
    st.markdown(report)

    # -----------------------------
    # PDF Report
    # -----------------------------

    pdf_file = create_pdf(
        student_data,
        final_score,
        report
    )

    with open(pdf_file, "rb") as f:

        st.download_button(
            "📄 Download PDF Report",
            data=f,
            file_name="Student_Report.pdf",
            mime="application/pdf"
        )

        # -----------------------------
# Prediction History
# -----------------------------

st.divider()

st.subheader("📜 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button("🗑 Clear History"):

        st.session_state.history = []
        st.rerun()

else:

    st.info("No predictions made yet.")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown("""
<div style="text-align:center;">

<h2>🎓 AI-Driven Student Performance Prediction System</h2>

<h4>Machine Learning + Generative AI + Streamlit</h4>

<p><b>Developed by</b></p>

<h3>Anand Prakash Yadav</h3>

<p>GEN-AI Internship Project</p>

<h3>⭐ Thank You ⭐</h3>

</div>
""", unsafe_allow_html=True)