import os
import json
from flask import Flask, render_template, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def analyze_cv_with_gemini(cv_text, job_description):
    prompt = f"""
You are an ATS and recruiter assistant.

Compare the candidate CV with the job description.

Return only valid JSON in exactly this structure:
{{
  "match_percentage": 0,
  "summary": "short summary",
  "strengths": ["item1", "item2", "item3"],
  "missing_skills": ["item1", "item2", "item3"],
  "suggestions": ["item1", "item2", "item3"]
}}

Rules:
- match_percentage must be an integer between 0 and 100
- be realistic and strict
- summary must be short
- strengths should mention relevant matching points
- missing_skills should mention actual gaps
- suggestions should be practical

Candidate CV:
{cv_text}

Job Description:
{job_description}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    raw_text = response.text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = {
            "match_percentage": 0,
            "summary": "AI response could not be parsed.",
            "strengths": [],
            "missing_skills": [],
            "suggestions": [raw_text]
        }

    return data


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        cv_file = request.files.get("cv_file")

        if not job_description or not cv_file:
            return render_template("index.html", error="Please upload a CV and enter a job description.")

        if not cv_file.filename.lower().endswith(".pdf"):
            return render_template("index.html", error="Please upload a PDF file only.")

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], cv_file.filename)
        cv_file.save(file_path)

        cv_text = extract_text_from_pdf(file_path)

        if not cv_text:
            return render_template("index.html", error="Could not extract text from the PDF.")

        result = analyze_cv_with_gemini(cv_text, job_description)
        return render_template("result.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)