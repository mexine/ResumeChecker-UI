import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import PyPDF2
from docx import Document

# Define a function to extract text from PDF files
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Define a function to extract text from DOCX files
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Define a simple text preprocessing function
def preprocess_text(text, noise_words=['n a', 'company name', 'city', 'state']):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    
    if noise_words:
        for word in noise_words:
            text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove noise words
    
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    return text

# Path to the directory containing resumes
resume_dir = 'C:\\Users\\justi\\Desktop\\DesignedFlask\\ResumeChecker-UI-main\\ResumeDownloads'

# Read text from each PDF and DOCX file in the directory along with filenames
resumes = []
resume_filenames = []
for filename in os.listdir(resume_dir):
    file_path = os.path.join(resume_dir, filename)
    if filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file_path)
        resumes.append(resume_text)
        resume_filenames.append(filename)
    elif filename.endswith('.docx'):
        resume_text = extract_text_from_docx(file_path)
        resumes.append(resume_text)
        resume_filenames.append(filename)

# Create a CountVectorizer to convert text to BoW representation
vectorizer = CountVectorizer(max_features=1000)  

# Fit and transform job description and resumes to BoW representation
job_description = "Maxine Claire T. TejadaAddress: Bogtong Bolo, Mangatarem, PangasinanContact No: +639-457-885-540Email Address: maxineclairetejada@gmail.comwww.linkedin.com/in/maxine-claire-tejada-a7482924"
job_desc_bow = vectorizer.fit_transform([job_description])
print("Jobdescbow", job_desc_bow)

resumes_preprocessed = [preprocess_text(resume) for resume in resumes]
resume_bow = vectorizer.transform(resumes_preprocessed)
print("Reusmebow", resume_bow)

# Calculate cosine similarity between job description and each resume
similarities = cosine_similarity(job_desc_bow, resume_bow)
print("Cosine similarity: ", similarities)

# Get the indices of the top 100 most similar resumes
top_similar_indices = similarities.argsort()[0][-100:][::-1]

# Print rankings of the top 100 most similar resumes along with filenames
print("\nTop 100 Most Similar Resumes:")
for rank, index in enumerate(top_similar_indices, start=1):
    similarity = similarities[0][index]
    print(f"Rank {rank}: Similarity (Resume {resume_filenames[index]}): {similarity:.4}\n")
