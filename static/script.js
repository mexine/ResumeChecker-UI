import { initializeApp } from "https://www.gstatic.com/firebasejs/10.6.0/firebase-app.js";
import { getStorage, ref, listAll, uploadBytes, deleteObject } from "https://www.gstatic.com/firebasejs/10.6.0/firebase-storage.js";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBVsmsAJNwokMv84gOIXjIPjbeCna63_9M",
  authDomain: "resume-checker-2.firebaseapp.com",
  projectId: "resume-checker-2",
  storageBucket: "resume-checker-2.appspot.com",
  messagingSenderId: "654874221520",
  appId: "1:654874221520:web:703d1e763d5710784177d5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

// Function to check if a file is a PDF or DOCX
function isFileAllowed(file) {
  const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']; // Mime types for PDF and DOCX
  return allowedTypes.includes(file.type);
}

// Function to delete selected files using checkbox
// async function deleteSelectedFiles(teFolderRef, bowFolderRef) {
//   const checkboxsTE = document.querySelectorAll('#fileTable .deleteCheckbox');
//   const checkboxsBOW = document.querySelectorAll('#fileTable1 .deleteCheckbox');

//   checkboxsTE.forEach(async (checkbox, index) => {
//     if(checkbox.checked) {
//       const row = checkbox.parentNode.parentNode;
//       const fileName = row.cells[1].textContent.trim();

//       try {
//         // Delete from TE folder
//         const teFileRef = ref(teFolderRef, fileName);
//         await deleteObject(teFileRef);
//         console.log('File $(fileName) deleted from TE folder');
//       } catch (error) {
//         console.error('Error');
//       }

//       // Remove from TE table
//       row.parentNode.removeChild(row);
//     }

//   })

//   checkboxsBOW.forEach(async (checkbox, index) => {
//     if(checkbox.checked) {
//       const row = checkbox.parentNode.parentNode;
//       const fileName = row.cells[1].textContent.trim();

//       try {
//         // Delete from BOW folder
//         const bowFileRef = ref(bowFolderRef, fileName);
//         await deleteObject(bowFileRef);
//         console.log('File $(fileName) deleted from BOW folder');
//       } catch (error) {
//         console.error('Error');
//       }

//       // Remove from BOW table
//       row.parentNode.removeChild(row);
//     }

//   })

//   // Reset the Select All checkbox after deletion
//   const selectAllTECheckbox = document.getElementById('selectAllTE');
//   const selectAllBOWCheckbox = document.getElementById('selectAllBOW');

//   selectAllTECheckbox.checked = false;
//   selectAllBOWCheckbox.checked = false;
// }

// Function to delete selected files using checkbox
async function deleteSelectedFiles(teFolderRef, bowFolderRef) {
  const checkboxesTE = document.querySelectorAll('#fileTable .deleteCheckbox');
  const checkboxesBOW = document.querySelectorAll('#fileTable1 .deleteCheckbox');

  checkboxesTE.forEach(async (checkbox) => {
    if (checkbox.checked) {
      const row = checkbox.parentNode.parentNode;
      const fileName = row.cells[1].textContent.trim();

      try {
        // Delete from TE folder
        const teFileRef = ref(teFolderRef, fileName);
        await deleteObject(teFileRef);
        console.log(`File ${fileName} deleted from TE folder`);
      } catch (error) {
        console.error('Error deleting file from TE folder:', error);
      }

      // Remove from TE table
      row.parentNode.removeChild(row);
    }
  });

  checkboxesBOW.forEach(async (checkbox) => {
    if (checkbox.checked) {
      const row = checkbox.parentNode.parentNode;
      const fileName = row.cells[1].textContent.trim();

      try {
        // Delete from BOW folder
        const bowFileRef = ref(bowFolderRef, fileName);
        await deleteObject(bowFileRef);
        console.log(`File ${fileName} deleted from BOW folder`);
      } catch (error) {
        console.error('Error deleting file from BOW folder:', error);
      }

      // Remove from BOW table
      row.parentNode.removeChild(row);
    }
  });

  // Uncheck the "Select All" checkboxes after deletion
  const selectAllTECheckbox = document.getElementById('selectAllTE');
  const selectAllBOWCheckbox = document.getElementById('selectAllBOW');

  selectAllTECheckbox.checked = false;
  selectAllBOWCheckbox.checked = false;
}

async function uploadFiles(files) {
  alert("Starting upload of resumes.")
  const teFolderRef = ref(storage, 'TEfolder/');
  const bowFolderRef = ref(storage, 'BoWfolder/');

  const listTeResult = await listAll(teFolderRef);
  const listBowResult = await listAll(bowFolderRef);

  const existingTeFiles = listTeResult.items.map(file => file.name);
  const existingBowFiles = listBowResult.items.map(file => file.name);

  const invalidFiles = [];
  const duplicateTeFiles = [];
  const duplicateBowFiles = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const fileName = file.name;

    console.log('uploading file: ' + i + ' ' + fileName)


    if (!isFileAllowed(file)) {
      invalidFiles.push(fileName);
      continue; // Skip uploading invalid file
    }

    //For TE table
    if (existingTeFiles.includes(fileName)) {
      duplicateTeFiles.push(fileName);
    } else {
      const teFileRef = ref(teFolderRef, fileName);
      try {
        await uploadBytes(teFileRef, file);
      } catch (error) {
        console.error('Error uploading file to TEfolder:', error);
        invalidFiles.push(fileName);
        continue; // Skip uploading invalid file
      }
    }

    // For BoW table
    if (existingBowFiles.includes(fileName)) {
      duplicateBowFiles.push(fileName);
    } else {
      const bowFileRef = ref(bowFolderRef, fileName);
      try {
        await uploadBytes(bowFileRef, file);
      } catch (error) {
        console.error('Error uploading file to BoWfolder:', error);
        invalidFiles.push(fileName);
        continue; // Skip uploading invalid file
      }
    }

  }

  // Display alert for duplicate files in TEfolder, if any
  if (duplicateTeFiles.length > 0) {
    alert(`Duplicate files detected in TEfolder: ${duplicateTeFiles.join(', ')}`);
  }

  // Display alert for duplicate files in BoWfolder, if any
  if (duplicateBowFiles.length > 0) {
    alert(`Duplicate files detected in BoWfolder: ${duplicateBowFiles.join(', ')}`);
  }

  // Display alert for invalid files, if any
  if (invalidFiles.length > 0) {
    alert(`Invalid files detected: ${invalidFiles.join(', ')}`);
  }

  fetch('/datasetprocess')
      .then(response => {
        console.log('local CSV files are updated!')
      })
      .catch(error => {
        console.error('Error:', error);
      });

  // Refresh the displayed files in the table
  displayFilesInTable(teFolderRef, bowFolderRef);
}

async function displayFilesInTable(teFolderRef, bowFolderRef) {
  const listTeResult = await listAll(teFolderRef);
  const listBowResult = await listAll(bowFolderRef);
  const teFiles = listTeResult.items;
  const bowFiles = listBowResult.items;

  const tableBody = document.querySelector('#fileTable tbody');
  const tableBody1 = document.querySelector('#fileTable1 tbody');

  tableBody.innerHTML = ''; // Clear the existing table content
  tableBody1.innerHTML = ''; // Clear the existing table content

  // Display files from TEfolder
  teFiles.forEach((file) => {
    displayFileInTable(file, tableBody);
  });

  // Display files from BoWfolder
  bowFiles.forEach((file) => {
    displayFileInTable(file, tableBody1);
  });

  fetch('/datasetprocess')
    .then(response => {
      console.log('local CSV files are updated!')
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

function displayFileInTable(file, tableBody) {
  const fileName = file.name;

  // Second column for File Name (as hyperlink)
  const fileNameCell = document.createElement('td');
  const fileLink = document.createElement('a');
  fileLink.textContent = fileName;
  const storagePath = encodeURIComponent(file.fullPath);
  const downloadURL = `https://firebasestorage.googleapis.com/v0/b/resumechecker-76d41.appspot.com/o/${storagePath}?alt=media`;
  fileLink.href = downloadURL;
  fileLink.target = "_blank"; // Open link in a new tab
  fileNameCell.appendChild(fileLink);

  // Create table row and cells
  const fileRow = document.createElement('tr');
  const rankCell = document.createElement('td');
  const qualificationCell = document.createElement('td');

  // Append cells to the row
  fileRow.appendChild(rankCell);
  fileRow.appendChild(fileNameCell);
  fileRow.appendChild(qualificationCell);

  // Append row to the table body
  tableBody.appendChild(fileRow);

  // Create a checkbox for each row
  const checkboxCell = document.createElement('td');
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.className = 'deleteCheckbox';
  checkboxCell.appendChild(checkbox);
  fileRow.appendChild(checkboxCell);

}

// Assuming you have storage references for TEfolder and BoWfolder
const teFolderRef = ref(storage, 'TEfolder/');
const bowFolderRef = ref(storage, 'BoWfolder/');

// Call the function to display files in the tables
displayFilesInTable(teFolderRef, bowFolderRef);


// Event listener for file input change
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', function () {
  const files = fileInput.files;
  uploadFiles(files);
});

// Event listener for upload button
const uploadButton = document.querySelector('.uploadButton');
uploadButton.addEventListener('click', function () {
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
clearButton.addEventListener('click', function () {
  // Set the value of the textarea to an empty string to clear its content
  jobDescriptionTextArea.value = '';
});

// // Event listener for Select All (TE) checkbox
// const selectAllTECheckbox = document.getElementById('selectAllTE');
// selectAllTECheckbox.addEventListener('change', function () {
//   const checkboxesTE = document.querySelectorAll('#fileTable .deleteCheckbox');
//   checkboxesTE.forEach((checkbox) => {
//     checkbox.checked = selectAllTECheckbox.checked;
//   });
// });

// // Event listener for Select All (BOW) checkbox
// const selectAllBOWCheckbox = document.getElementById('selectAllBOW');
// selectAllBOWCheckbox.addEventListener('change', function () {
//   const checkboxesBOW = document.querySelectorAll('#fileTable1 .deleteCheckbox');
//   checkboxesBOW.forEach((checkbox) => {
//     checkbox.checked = selectAllBOWCheckbox.checked;
//   });
// });

// // Deletion of files 
// const tableClearbtn = document.querySelector('.tableClearButton');
// tableClearbtn.addEventListener('click', function(){
//   deleteSelectedFiles(teFolderRef, bowFolderRef);
//   // update csv files

//   fetch('/datasetprocess')
//     .then(response => {
//       console.log('local CSV files are updated!')
//     })
//     .catch(error => {
//       console.error('Error:', error);
//     });
// });

// Event listener for Clear button to delete selected files
const tableClearButton = document.querySelector('.tableClearButton');
tableClearButton.addEventListener('click', function () {
  console.log('tableClearButton')
  deleteSelectedFiles(teFolderRef, bowFolderRef);

  fetch('/datasetprocess')
  .then(response => {
    console.log('local CSV files are updated!')
  })
  .catch(error => {
    console.error('Error:', error);
  });
});

// Function to handle "Select All" for checkboxes
function handleSelectAll(checkbox, tableId) {
  const checkboxes = document.querySelectorAll(`#${tableId} .deleteCheckbox`);
  checkboxes.forEach((chkBox) => {
    chkBox.checked = checkbox.checked;
  });
}

// Event listener for "Select All" checkbox in TE table
const selectAllTECheckbox = document.getElementById('selectAllTE');
selectAllTECheckbox.addEventListener('change', function () {
  handleSelectAll(selectAllTECheckbox, 'fileTable');
});

// Event listener for "Select All" checkbox in BOW table
const selectAllBOWCheckbox = document.getElementById('selectAllBOW');
selectAllBOWCheckbox.addEventListener('change', function () {
  handleSelectAll(selectAllBOWCheckbox, 'fileTable1');
});

async function analyzeBoW() {
  try {
    const jobDescription = document.getElementById('jobDescription').value; // Get job description string from textarea
   
    if (jobDescription === '') {
      alert('Analysis cannot proceed without a job description.');
      return;
    }

    const response = await fetch('/analyze_bow', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ jobDescription: jobDescription })
    });

    const data = await response.json(); // Parse JSON response

    const tableBody = document.querySelector('#fileTable1 tbody');
    tableBody.innerHTML = ''; // Clear any existing rows in the table

    data.results.forEach(item => {
      const resumeLink = `${item.URL}`;
      const newRow = `
        <tr>
          <td>${item.Rank}</td>
          <td><a href="${resumeLink}" target="_blank">${item.Filename}</a></td>
          <td>${item.Similarity}</td>
          <td><input type="checkbox"></td>
        </tr>
      `;
      tableBody.insertAdjacentHTML('beforeend', newRow);
    });
    alert("FinishedBow");
  } catch (error) {
    console.error('Error:', error);
  }
}

async function analyzeTE() {
  console.log('analyzeTE')
  try {
    const jobDescription = document.getElementById('jobDescription').value; // Get job description string from textarea
    console.log(jobDescription)
    const response = await fetch('/analyze_te', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ jobDescription: jobDescription })
    });

    const data = await response.json(); // Parse JSON response

    const tableBody = document.querySelector('#fileTable tbody');
    tableBody.innerHTML = ''; // Clear any existing rows in the table

    data.results.forEach(item => {
      const resumeLink = `${item.URL}`;
      const newRow = `
        <tr>
          <td>${item.Rank}</td>
          <td><a href="${resumeLink}" target="_blank">${item.Filename}</a></td>
          <td>${item.Similarity}</td>
          <td><input type="checkbox"></td>
        </tr>
      `;
      tableBody.insertAdjacentHTML('beforeend', newRow);
    });
    alert("FinishedTE");
  } catch (error) {
    console.error('Error:', error);
  }
}

// Event listener for the "Analyze" button to trigger the download of all files
const analyzeButton = document.querySelector('.analyzeButton');
analyzeButton.addEventListener('click', function () {
  alert("Reading resumes...");
  fetch('/datasetprocess')
    .then(response => {
      analyzeBoW(); // Analysis for table-side BoW
      analyzeTE(); 
    })
    .catch(error => {
      console.error('Error:', error);
    });
});

