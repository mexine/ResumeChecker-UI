import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Define a simple text preprocessing function
def preprocess_text(text, noise_words=['n a', 'company name', 'city', 'state']):
    text = str(text)  # Convert to string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    
    if noise_words:
        for word in noise_words:
            text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove noise words
    
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    return text

# Load the CSV file containing resume filenames and contents
resume_data = pd.read_csv("resume_contents.csv")

# Job description
job_description = "Maxine Claire T. Tejada, Address: Bogtong Bolo, Mangatarem, Pangasinan, Contact No: +639-457-885-540, Email Address: maxineclairetejada@gmail.com, www.linkedin.com/in/maxine-claire-tejada-a74829247, Career Objective: Junior computer science student at Polytechnic University of the Philippines seeking an internship. Keen to earn a practicum where I can utilize my technical expertise and problem-solving skills to further expand my abilities in the field of technology. Education: Polytechnic University of the Philippines, Tertiary Education 2020 - Current, Consistent President’s Lister, Bachelor of Science in Computer Science. University of Pangasinan, Upper Secondary Education 2018-2020, With Honors, Science Technology Engineering and Mathematics. Mangatarem National High School, Lower Secondary Education 2014-2018, With Honors, Science Technology Engineering and Mathematics. Skills: Programming languages - C++, C, Java, and Python, Web Development Technologies - HTML, CSS, and JavaScript, Technologies - Windows, Android, Git, MySQL, General - Teamwork, Communication, Organization, Time management, Problem-solving, Creative thinking. Projects: Listahan: To-do List Application - Co-creator, Used Java and MySQL in creation and implementation of the app. Implemented Quicksort and Linear Search Algorithm. Summarizer - Co-creator, Coded in Java, An extractive summarizer using NLP. FoodBuddies - Co-creator, Food blog website. PUPKeep: Maintenance Reporting Application - Co-creator, Centralized platform for reporting incidents and streamlining tasks for maintenance personnel."

# Extract resume filenames and contents from the loaded data
resume_filenames = resume_data['Filename'].tolist()
resumes = resume_data['Contents'].tolist()

# Create a CountVectorizer to convert text to BoW representation
vectorizer = CountVectorizer(max_features=1000)  

# Fit and transform job description and resumes to BoW representation
job_desc_bow = vectorizer.fit_transform([job_description])
resumes_preprocessed = [preprocess_text(resume) for resume in resumes]
resume_bow = vectorizer.transform(resumes_preprocessed)

# Calculate cosine similarity between job description and each resume
similarities = cosine_similarity(job_desc_bow, resume_bow)
print("Cosine similarity: ", similarities)

# Rank and print the resumes based on cosine similarity
ranked_resumes = []
for index, (filename, similarity) in enumerate(zip(resume_filenames, similarities[0]), start=1):
    ranked_resumes.append((filename, similarity))

# Sort the ranked resumes based on similarity score in descending order
ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

# Print the rankings of resumes with their filenames and cosine similarity
print("\nRanked Resumes:")
for rank, (filename, similarity) in enumerate(ranked_resumes[:100], start=1):
    print(f"Rank {rank}: Filename: {filename}, Similarity: {similarity:.4}\n")