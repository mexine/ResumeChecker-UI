import csv
from firebase_admin import storage
import fitz  # PyMuPDF for PDF handling
from docx import Document  # python-docx for DOCX handling
import firebase_admin
import io
import urllib.parse

# Initialize Firebase Admin SDK
cred = firebase_admin.credentials.Certificate(r'resumechecker-76d41-firebase-adminsdk-fyaue-4bb05d5713.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'resumechecker-76d41.appspot.com'})

# Reference to Firebase Storage bucket
bucket = storage.bucket(name='resumechecker-76d41.appspot.com')

# Function to read the contents of a DOCX file
def read_docx_content(blob):
    doc = Document(io.BytesIO(blob.download_as_bytes()))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to read the contents of a PDF file
def read_pdf_content(blob):
    pdf_content = io.BytesIO(blob.download_as_bytes())
    pdf_document = fitz.open(stream=pdf_content.read())
    full_text = ''
    for page in pdf_document:
        full_text += page.get_text()
    return full_text

# Function to generate download URL with manually inserted %2F
def generate_download_url(file_name):
    parts = file_name.split('/')
    folder_name = parts[-2]  # Extract the folder name
    file_path = urllib.parse.quote(f'{folder_name}/{parts[-1]}', safe='')
    return f"https://firebasestorage.googleapis.com/v0/b/resumechecker-76d41.appspot.com/o/{file_path}?alt=media"

# Get a list of files from the storage bucket
blobs = bucket.list_blobs()
files = [blob for blob in blobs if blob.name.endswith('.pdf') or blob.name.endswith('.docx')]

data = {'TEfolder': [], 'BoWfolder': []}  # Separate data for different folders

# Loop through files and extract content
for blob in files:
    filename = blob.name.split('/')[-1]
    folder_name = blob.name.split('/')[-2]

    if blob.name.endswith('.pdf'):
        file_content = read_pdf_content(blob)
        if not file_content:
            file_content = '&&PLACEHOLDER&&'  # Replace with literal placeholder text to prevent TE from locking.

    elif blob.name.endswith('.docx'):
        file_content = read_docx_content(blob)
        if not file_content:
            file_content = '&&PLACEHOLDER&&'  

    file_url = generate_download_url(blob.name)
    data[folder_name].append([filename, file_content, file_url])


# Function to write data to CSV file
def write_to_csv(file_name, dataset):
    csv_file_path = f'resume_contents_{file_name}.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Contents', 'File_URL'])  # Writing headers
        csv_writer.writerows(dataset)

    print(f"CSV file '{csv_file_path}' has been created with the resumes from '{file_name}' folder.")

# Write data to CSV files for each folder
for folder_name, folder_data in data.items():
    write_to_csv(folder_name, folder_data)    
