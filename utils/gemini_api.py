import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

# Gemini Client
client = genai.Client(api_key=api_key)


def generate_ai_report(student_data, predicted_score):

    prompt = f"""
You are an educational AI assistant.

Student Information:
Age: {student_data['Age']}
Study Hours: {student_data['Study_Hours']}
Attendance: {student_data['Attendance']}
Previous Score: {student_data['Previous_Score']}
Assignment Score: {student_data['Assignment_Score']}
Sleep Hours: {student_data['Sleep_Hours']}
Internet Usage: {student_data['Internet_Usage']}
Participation: {student_data['Participation']}

Predicted Final Score: {predicted_score}

Generate a report with these headings:

1. Overall Performance
2. Strengths
3. Weaknesses
4. Suggestions

Keep the response short (150–200 words).
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",   # <-- Change this line
            contents=prompt
        )

        if hasattr(response, "text") and response.text:
            return response.text
        else:
            return "No response generated."

    except Exception as e:
        return f"❌ Gemini Error:\n{str(e)}"