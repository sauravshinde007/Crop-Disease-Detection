import React from "react";

const CropSelector = ({ selectedCrop, setSelectedCrop }) => {
  const crops = ["Cashew", "Cassava", "Maize", "Tomato"];

  return (
    <div className="mb-4">
      <label className="block text-lg font-semibold text-gray-800">Select Crop:</label>
      <select
        className="mt-2 p-3 border rounded-lg w-full focus:ring-2 focus:ring-green-500 transition duration-200"
        value={selectedCrop}
        onChange={(e) => setSelectedCrop(e.target.value)}
      >
        <option value="">-- Choose Crop --</option>
        {crops.map((crop) => (
          <option key={crop} value={crop}>
            {crop}
          </option>
        ))}
      </select>
    </div>
  );
};

export default CropSelector;
