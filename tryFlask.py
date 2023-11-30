from flask import Flask, render_template, request, jsonify, send_from_directory

import firebase_admin
from firebase_admin import storage
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import subprocess

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download stopwords from nltk here
import nltk
nltk.download('stopwords')

import re
import os

from transformer_encoder.tokenizer import Tokenizer
from transformer_encoder.base import Sequential
from transformer_encoder.layers import WordEmbedding, PositionalEncoding, Dense, MultiHeadAttention, SelfAttention, LayerNormalization
from transformer_encoder.activation import ReLu, Linear, Softmax

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = firebase_admin.credentials.Certificate(r'resumechecker-76d41-firebase-adminsdk-fyaue-4bb05d5713.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'resumechecker-76d41.appspot.com'})

# Function to preprocess text
def preprocess_text(text, noise_words=['n a', 'company name', 'city', 'state']):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    
    # NLTK stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    # Snowball Stemmer
    stemmer = SnowballStemmer('english')
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    
    if noise_words:
        for word in noise_words:
            text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove noise words
    
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    return text


@app.route('/')
def index():
    print('test index.html')
    return render_template('index.html') 

    
@app.route('/datasetprocess', methods=['GET'])
def trigger_script():
    try:
        # Run the toCsv.py script externally using Python's subprocess module
        subprocess.run(['python', 'toCsv.py'])
        return 'Script triggered successfully!'
    except Exception as e:
        return f'Error triggering script: {str(e)}'

# Route for analyzing job description against resumes using Bag of Words
@app.route('/analyze_bow', methods=['POST'])
def analyze_bow():
    try:
        logging.info("hi bow")
        # Load resume data from CSV
        resume_data = pd.read_csv("resume_contents_BoWfolder.csv")
        
        # Get the job description from the form
        job_description = preprocess_text(request.json.get('jobDescription'))
        # print(job_description)

        # Extract resume filenames and contents
        resume_filenames = resume_data['Filename'].tolist()
        resumes = resume_data['Contents'].tolist()
        resume_urls = resume_data['File_URL'].tolist()
        
        # Create a CountVectorizer to convert text to BoW representation
        vectorizer = CountVectorizer(max_features=1000)     

        # Fit and transform job description and resumes to BoW representation
        job_desc_bow = vectorizer.fit_transform([job_description])
        resumes_preprocessed = [preprocess_text(resume) for resume in resumes]
        resume_bow = vectorizer.transform(resumes_preprocessed)

        # Calculate cosine similarity between job description and each resume
        similarities = cosine_similarity(job_desc_bow, resume_bow)
        # print("Similarities Matrix:")
        # print(similarities)

        ranked_resumes_data = []
        
        # Create a list of tuples (index, similarity) for sorting
        similarity_scores = [(idx, score) for idx, score in enumerate(similarities[0])]
        
        # Sort similarity scores in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (idx, similarity) in enumerate(similarity_scores, start=1):
            filename = resume_filenames[idx]
            url = resume_urls[idx]
            similarity_percentage = round(similarity * 100, 2)  # Convert similarity to percentage
            ranked_resumes_data.append({
                "Rank": rank,
                "Filename": filename,
                "URL": url,  # Include Firebase URL in the response
                "Similarity": f"{similarity_percentage}%"  # Format similarity as percentage
            })

        return jsonify(results=ranked_resumes_data)

    except Exception as e:
        return jsonify(error=str(e))
    

# Route for analyzing job description against resumes using the Transformer Encoder
@app.route('/analyze_te', methods=['POST'])
def analyze_te():
    try:
        logging.info('/analyze_te was accessed.')

        # Load resume data from CSV
        resume_data = pd.read_csv("resume_contents_TEfolder.csv")
        
        # Get the job description from the form
        job_description = request.json.get('jobDescription')

        logging.info(job_description)

        # Extract resume filenames and contents
        resume_filenames = resume_data['Filename'].tolist()

        logging.info('resume filenames' + str(len(resume_filenames)))

        resumes = resume_data['Contents'].tolist()

        logging.info('resumes ' + str(len(resumes)))

        resume_urls = resume_data['File_URL'].tolist()


        # # Create a CountVectorizer to convert text to BoW representation
        # vectorizer = CountVectorizer(max_features=1000)     

        # # Fit and transform job description and resumes to BoW representation
        # job_desc_bow = vectorizer.fit_transform([job_description])
        # resumes_preprocessed = [preprocess_text(resume) for resume in resumes]
        # resume_bow = vectorizer.transform(resumes_preprocessed)

        # hyperparameters
        vocab_size = 22789
        model_dim = 512
        num_heads = 8
        ffn_dim = 2048
        max_pos = 512

        tokenizer = Tokenizer(max_pos=max_pos, vocab_size=vocab_size)

        logging.info(tokenizer)

        tokenizer.load_word_index()

        logging.info(tokenizer)

        # instantiate the model
        model = Sequential([
                    WordEmbedding(vocab_size, model_dim),
                    PositionalEncoding(max_pos, model_dim),
                    MultiHeadAttention(num_heads, max_pos, model_dim),
                    # SelfAttention(max_pos, model_dim),
                    LayerNormalization(model_dim),
                    # Feed Forward Network
                    Dense([model_dim, ffn_dim], ReLu),
                    Dense([ffn_dim, model_dim], Linear),
                    LayerNormalization(model_dim),
                    # # MLM Head
                    # Dense([model_dim, vocab_size], Softmax)
        ])

        # model.load_model()

        logging.info(model)

        preprocessed_job_desc, attention_mask = tokenizer.clean_truncate_tokenize_pad_atten(job_description)

        logging.info(preprocessed_job_desc.shape)

        job_desc_te = model.predict(preprocessed_job_desc[0], attention_mask[0])

        logging.info(job_desc_te.shape)

        resumes_te = []
        for index, resume in enumerate(resumes):
            preprocessed_resume, attention_mask = tokenizer.clean_truncate_tokenize_pad_atten(resume)
            resumes_te.append(model.predict(preprocessed_resume[0], attention_mask[0]))
            logging.info(str(index))

        # preprocessed_resume, attention_mask = tokenizer.clean_truncate_tokenize_pad_atten(resumes[0])
        # resumes_te = model.predict(preprocessed_resume[0], attention_mask[0])

        # Flatten the arrays
        job_desc_te = job_desc_te.flatten()
        job_desc_te = job_desc_te.reshape(1, -1)

        # Calculate cosine similarity between job description and each resume
        similarities = []
        for resumes_te_embedding in resumes_te:
            resumes_te_embedding = resumes_te_embedding.flatten()
            resumes_te_embedding = resumes_te_embedding.reshape(1, -1)
            similarities.append(float(cosine_similarity(job_desc_te, resumes_te_embedding)[0][0]))
            logging.info(similarities)

        logging.info(similarities)

        # Rank resumes based on similarity score (integrating logic from AnalyzeBoW)
        ranked_resumes_data = []
        similarity_scores = [(idx, score) for idx, score in enumerate(similarities)]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (idx, similarity) in enumerate(similarity_scores, start=1):
            filename = resume_filenames[idx]
            url = resume_urls[idx]  # Retrieving URL instead of file path
            similarity_percentage = round(similarity * 100, 2)  # Convert similarity to percentage

            ranked_resumes_data.append({
                "Rank": rank,
                "Filename": filename,
                "URL": url,  # Include URL in the response similar to AnalyzeBoW
                "Similarity": f"{similarity_percentage}%"  # Format similarity as percentage
            })

        # Return the ranked resumes data as JSON using Flask's jsonify
        return jsonify(results=ranked_resumes_data)

    except Exception as e:
        return jsonify(error=str(e))

    
if __name__ == '__main__':
    app.run(debug=True)
