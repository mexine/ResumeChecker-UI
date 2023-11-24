// Import the functions you need from the SDKs you need
// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.6.0/firebase-app.js";
import { getStorage, ref, listAll, uploadBytes } from "https://www.gstatic.com/firebasejs/10.6.0/firebase-storage.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBRdp6TQsoPo4PDkMPQErqg2GMmCLwigUQ",
  authDomain: "resumechecker-76d41.firebaseapp.com",
  projectId: "resumechecker-76d41",
  storageBucket: "resumechecker-76d41.appspot.com",
  messagingSenderId: "509510006446",
  appId: "1:509510006446:web:ebbf01eb9e9713f0bd1f3f"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

// Function to check if a file is a PDF or DOCX
function isFileAllowed(file) {
  const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']; // Mime types for PDF and DOCX
  return allowedTypes.includes(file.type);
}

// Function to upload files
async function uploadFiles(files) {
  const storageRef = ref(storage, 'resumes/');
  const listResult = await listAll(storageRef);
  const existingFiles = listResult.items.map(file => file.name);

  const invalidFiles = [];
  const duplicateFiles = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const fileName = file.name;

    if (!isFileAllowed(file)) {
      invalidFiles.push(fileName);
      continue; // Skip uploading invalid file
    }

    if (existingFiles.includes(fileName)) {
      duplicateFiles.push(fileName);
    } else {
      const storageFileRef = ref(storage, `resumes/${fileName}`);
      try {
        await uploadBytes(storageFileRef, file);
      } catch (error) {
        console.error('Error uploading file:', error);
        invalidFiles.push(fileName);
        continue; // Skip uploading invalid file
      }
    }
  }

  // Display alert for duplicate files, if any
  if (duplicateFiles.length > 0) {
    alert(`Duplicate files detected: ${duplicateFiles.join(', ')}`);
  }

  // Display alert for invalid files, if any
  if (invalidFiles.length > 0) {
    alert(`Invalid files detected: ${invalidFiles.join(', ')}`);
  }

  // Refresh the displayed files in the table
  displayFilesInTable();
}


async function displayFilesInTable() {
  const storageRef = ref(storage, 'resumes/');
  const listResult = await listAll(storageRef);
  const files = listResult.items;

  const tableBody = document.querySelector('#fileTable1 tbody');
  tableBody.innerHTML = ''; // Clear the existing table content

  files.forEach((file) => {
    const fileName = file.name;

    const fileRow = document.createElement('tr');

    // First column for Rank 
    const rankCell = document.createElement('td');
    fileRow.appendChild(rankCell);

    // Second column for File Name (as hyperlink)
    const fileNameCell = document.createElement('td');
    const fileLink = document.createElement('a');
    fileLink.textContent = fileName;
    const storagePath = encodeURIComponent(file.fullPath);
    const downloadURL = `https://firebasestorage.googleapis.com/v0/b/resumechecker-76d41.appspot.com/o/${storagePath}?alt=media`;
    fileLink.href = downloadURL;
    fileLink.target = "_blank"; // Open link in a new tab
    fileNameCell.appendChild(fileLink);
    fileRow.appendChild(fileNameCell);

    // Third column for Qualification 
    const qualificationCell = document.createElement('td');
    fileRow.appendChild(qualificationCell);

    tableBody.appendChild(fileRow);
  });
}

// Call the function to display files in the table
displayFilesInTable();

// Event listener for file input change
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', function() {
  const files = fileInput.files;
  uploadFiles(files);
});

// Event listener for upload button
const uploadButton = document.querySelector('.uploadButton');
uploadButton.addEventListener('click', function() {
  fileInput.click();
});

// Drag and drop functionality
const uploadContainer = document.getElementById('uploadContainer');

uploadContainer.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', () => {
  uploadContainer.classList.remove('dragover');
});

uploadContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadContainer.classList.remove('dragover');
  const droppedFiles = e.dataTransfer.files;
  uploadFiles(droppedFiles);
});

// Select the clear button and textarea element
const clearButton = document.querySelector('.clearButton');
const jobDescriptionTextArea = document.getElementById('jobDescription');

// Add a click event listener to the clear button
clearButton.addEventListener('click', function() {
  // Set the value of the textarea to an empty string to clear its content
  jobDescriptionTextArea.value = '';
});
