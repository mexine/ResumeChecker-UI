import os
import csv
import fitz  # PyMuPDF for PDF handling
from docx import Document  # python-docx for DOCX handling

directory = r'ResumeDownloads'

# Function to read the contents of a DOCX file
def read_docx_content(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to read the contents of a PDF file
def read_pdf_content(file_path):
    pdf_document = fitz.open(file_path)
    full_text = ''
    for page in pdf_document:
        full_text += page.get_text()
    return full_text

# Get a list of files in the directory
files = [file for file in os.listdir(directory) if file.endswith('.pdf') or file.endswith('.docx')]

# Prepare data to write to CSV
data = []
for file_name in files:
    file_path = os.path.join(directory, file_name)
    if file_name.endswith('.pdf'):
        file_content = read_pdf_content(file_path)
    elif file_name.endswith('.docx'):
        file_content = read_docx_content(file_path)
    data.append([file_name, file_content, directory])  # Append directory path

# Write data to a CSV file in the directory one step above
csv_file_path = r'resume_contents.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Filename', 'Contents', 'Directory'])  # Writing headers
    csv_writer.writerows(data)

print(f"CSV file '{csv_file_path}' has been generated.")
