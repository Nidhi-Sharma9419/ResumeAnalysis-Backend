from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_resume_text(resume_path):
    """Detect file type and extract text."""
    if resume_path.endswith('.pdf'):
        return extract_text_from_pdf(resume_path)
    elif resume_path.endswith('.docx'):
        return extract_text_from_docx(resume_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")