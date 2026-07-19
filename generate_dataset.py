# ==========================================
# AI Student Performance Prediction System
# Generate Student Dataset
# ==========================================

import pandas as pd
import numpy as np

# Same dataset every time
np.random.seed(42)

# Number of Students
num_students = 1000

# Generate Features
age = np.random.randint(17, 23, num_students)

study_hours = np.random.uniform(1, 8, num_students).round(1)

attendance = np.random.randint(60, 101, num_students)

previous_score = np.random.randint(40, 96, num_students)

assignment_score = np.random.randint(45, 101, num_students)

sleep_hours = np.random.uniform(4, 9, num_students).round(1)

internet_usage = np.random.uniform(1, 8, num_students).round(1)

participation = np.random.randint(1, 11, num_students)

# Final Score Formula
final_score = (
    previous_score * 0.35 +
    assignment_score * 0.20 +
    attendance * 0.20 +
    study_hours * 3 +
    participation * 2 -
    internet_usage * 1.5 +
    sleep_hours * 1.5 +
    np.random.normal(0, 3, num_students)
)

# Limit Score
final_score = np.clip(final_score, 0, 100).round().astype(int)

# Create DataFrame
dataset = pd.DataFrame({
    "Age": age,
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Score": previous_score,
    "Assignment_Score": assignment_score,
    "Sleep_Hours": sleep_hours,
    "Internet_Usage": internet_usage,
    "Participation": participation,
    "Final_Score": final_score
})

# Save CSV
dataset.to_csv("dataset.csv", index=False)

print("✅ Dataset Generated Successfully!")
print(dataset.head())
print("\nDataset Shape :", dataset.shape)