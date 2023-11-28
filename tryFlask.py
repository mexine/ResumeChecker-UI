from flask import Flask, render_template, request, jsonify, send_from_directory

import firebase_admin
from firebase_admin import storage
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = firebase_admin.credentials.Certificate(r'resumechecker-76d41-firebase-adminsdk-fyaue-4bb05d5713.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'resumechecker-76d41.appspot.com'})

# Function to preprocess text
def preprocess_text(text, noise_words=['n a', 'company name', 'city', 'state']):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    
    if noise_words:
        for word in noise_words:
            text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove noise words
    
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    return text

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/download_from_firebase', methods=['GET'])
def download_from_firebase():
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs()

        #target_folder = 'C:\\Users\\justi\\Desktop\\DesignedFlask\\ResumeChecker-UI-main\\ResumeDownloads'
        target_folder = 'ResumeDownloads'

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        for blob in blobs:
            file_name = blob.name.split('/')[-1]  # Get the file name from the blob's full path
            local_file_path = os.path.join(target_folder, file_name)
            blob.download_to_filename(local_file_path)

        return 'Files downloaded successfully!'
    except Exception as e:
        return f'Error downloading files: {str(e)}'
    
@app.route('/ResumeDownloads/<path:filename>')
def download_resume(filename):
    directory = 'C:\\Users\\justi\\Desktop\\DesignedFlask\\ResumeChecker-UI-main\\ResumeDownloads'
    return send_from_directory(directory, filename)

# Route for analyzing job description against resumes using Bag of Words
@app.route('/analyze_bow', methods=['POST'])
def analyze_bow():
    try:
        # Load resume data from CSV
        resume_data = pd.read_csv("resume_contents.csv")
        
        # Get the job description from the form
        # job_description = "Maxine Claire T. Tejada, Address: Bogtong Bolo, Mangatarem, Pangasinan, Contact No: +639-457-885-540, Email Address: maxineclairetejada@gmail.com, www.linkedin.com/in/maxine-claire-tejada-a74829247, Career Objective: Junior computer science student at Polytechnic University of the Philippines seeking an internship. Keen to earn a practicum where I can utilize my technical expertise and problem-solving skills to further expand my abilities in the field of technology. Education: Polytechnic University of the Philippines, Tertiary Education 2020 - Current, Consistent President’s Lister, Bachelor of Science in Computer Science. University of Pangasinan, Upper Secondary Education 2018-2020, With Honors, Science Technology Engineering and Mathematics. Mangatarem National High School, Lower Secondary Education 2014-2018, With Honors, Science Technology Engineering and Mathematics. Skills: Programming languages - C++, C, Java, and Python, Web Development Technologies - HTML, CSS, and JavaScript, Technologies - Windows, Android, Git, MySQL, General - Teamwork, Communication, Organization, Time management, Problem-solving, Creative thinking. Projects: Listahan: To-do List Application - Co-creator, Used Java and MySQL in creation and implementation of the app. Implemented Quicksort and Linear Search Algorithm. Summarizer - Co-creator, Coded in Java, An extractive summarizer using NLP. FoodBuddies - Co-creator, Food blog website. PUPKeep: Maintenance Reporting Application - Co-creator, Centralized platform for reporting incidents and streamlining tasks for maintenance personnel."
        job_description = request.json.get('jobDescription')
        preprocess_text(job_description)
        print(job_description)

        # Extract resume filenames and contents
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

        # Combine filenames, similarities, and links
        ranked_resumes = []
        for index, (filename, similarity) in enumerate(zip(resume_filenames, similarities[0]), start=1):
            ranked_resumes.append((filename, similarity))

        # Sort the ranked resumes based on similarity score in descending order
        ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

        ranked_resumes_data = []
        for rank, (filename, similarity) in enumerate(ranked_resumes[:100], start=1):
            file_path = os.path.join("ResumeDownloads", filename)  # File path
            similarity_percentage = round(similarity * 100, 2)  # Convert similarity to percentage
            ranked_resumes_data.append({
                "Rank": rank,
                "Filename": filename,
                "FilePath": file_path,  # Include file path in the response
                "Similarity": f"{similarity_percentage}%"  # Format similarity as percentage
            })

        # Return the ranked resumes data as JSON using Flask's jsonify
        return jsonify(results=ranked_resumes_data)

    except Exception as e:
        return jsonify(error=str(e))

    
if __name__ == '__main__':
    app.run(debug=True)
