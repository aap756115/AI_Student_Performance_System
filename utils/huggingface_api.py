import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_ai_report(student_data, predicted_score):

    prompt = f"""
You are an educational AI assistant.

Student Details:

Age: {student_data['Age']}
Study Hours: {student_data['Study_Hours']}
Attendance: {student_data['Attendance']}
Previous Score: {student_data['Previous_Score']}
Assignment Score: {student_data['Assignment_Score']}
Sleep Hours: {student_data['Sleep_Hours']}
Internet Usage: {student_data['Internet_Usage']}
Participation: {student_data['Participation']}

Predicted Final Score: {predicted_score}

Generate:

1. Overall Performance
2. Strengths
3. Weaknesses
4. Suggestions

Keep the response within 150 words.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]

        return str(result)

    except Exception as e:
        return f"❌ Hugging Face Error:\n{e}"