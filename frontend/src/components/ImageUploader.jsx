import React from "react";

const ImageUploader = ({ setImage, setPreview, preview }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  return (
    <div className="mb-4">
      <label className="block text-lg font-semibold text-gray-800">Upload Image:</label>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="w-full mt-2 p-2 border rounded-lg"
      />
      {preview && (
        <img
          src={preview}
          alt="Preview"
          className="mt-4 w-48 h-48 object-cover border-4 border-green-500 rounded-lg shadow-lg"
        />
      )}
    </div>
  );
};

export default ImageUploader;
