import os
from utils.extract_text import extract_resume_text
from utils.ollama_query import query_ollama

# Paths
INPUT_DIR = "input"
OUTPUT_DIR = "output"
RESUME_PATH = os.path.join(INPUT_DIR, "resume.pdf")  # Change to .docx if needed
JOB_DESC_PATH = os.path.join(INPUT_DIR, "job_description.txt")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "analysis_results.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_job_description(file_path):
    """Read job description from a text file."""
    with open(file_path, 'r') as file:
        return file.read()

def generate_analysis_prompt(resume_text, job_desc):
    """Generate a structured prompt for Ollama."""
    return f"""
    *Task:* Analyze this resume against the job description and provide actionable insights.

    *Resume:*
    {resume_text}

    *Job Description:*
    {job_desc}

    *Instructions:*
    1. List key skills that match the job requirements.
    2. Identify missing qualifications.
    3. Rate suitability (1-10) with justification.
    4. Suggest resume improvements.
    5. Recommend interview preparation tips.
    """

def main():
    # Step 1: Extract text from resume and job description
    resume_text = extract_resume_text(RESUME_PATH)
    job_desc = read_job_description(JOB_DESC_PATH)

    # Step 2: Generate the prompt
    prompt = generate_analysis_prompt(resume_text, job_desc)

    # Step 3: Query Ollama
    analysis_result = query_ollama(prompt)

    # Step 4: Save results
    with open(OUTPUT_PATH, 'w') as file:
        file.write(analysis_result)

    print(f"Analysis saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()