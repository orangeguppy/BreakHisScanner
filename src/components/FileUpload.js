import React, { useState } from "react";

function FileUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  console.log(selectedFile);

    const handleFileChange = (event) => {
      const file = event.target.files[0];
      const fileType = file.type;

      if (!fileType.startsWith('image/')) {
        alert('Please select an image file.');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(URL.createObjectURL(file));
  };

  return(
    <div class="card w-96 bg-base-100 shadow-xl">
      <figure class="px-10 pt-10">
        <input type="file" class="file-input w-full max-w-xs" accept="image/*" onChange={handleFileChange}/>
      </figure>
      <div class="card-body items-center text-center">
        <h2 class="card-title">Upload a scan from your device</h2>
        <p>The results from various CNNs will be displayed after processing.</p>
        <div class="card-actions">
          <button class="btn btn-primary">Submit for Scanning</button>
        </div>
      </div>
    </div>
  )
}export default FileUpload;
