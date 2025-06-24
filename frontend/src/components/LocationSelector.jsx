import React, { useState, useEffect } from "react";
import statesData from "../assets/states-and-districts.json"; // Adjust the path based on your file structure

const LocationSelector = ({ onLocationSelect }) => {
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [districts, setDistricts] = useState([]);

  // Update districts when state changes
  useEffect(() => {
    if (selectedState) {
      const stateObj = statesData.states.find(
        (state) => state.state === selectedState
      );
      setDistricts(stateObj ? stateObj.districts : []);
      setSelectedDistrict(""); // Reset district when state changes
    } else {
      setDistricts([]);
    }
  }, [selectedState]);

  // Handle selection changes
  const handleStateChange = (e) => {
    setSelectedState(e.target.value);
  };

  const handleDistrictChange = (e) => {
    setSelectedDistrict(e.target.value);
    onLocationSelect(selectedState, e.target.value); // Pass the selected location to parent component
  };

  return (
    <div className="mx-auto p-4 bg-white shadow rounded-lg w-full max-w-md">
      <h2 className="text-lg font-semibold mb-3">📍 Select Location</h2>

      {/* State Selection */}
      <label className="block text-lg font-medium text-gray-700">State:</label>
      <select
        className="mt-1 p-2 border rounded w-full"
        value={selectedState}
        onChange={handleStateChange}
      >
        <option value="">-- Choose State --</option>
        {statesData.states.map((state) => (
          <option key={state.state} value={state.state}>
            {state.state}
          </option>
        ))}
      </select>

      {/* District Selection */}
      {selectedState && (
        <>
          <label className="block text-lg font-medium text-gray-700 mt-3">
            District:
          </label>
          <select
            className="mt-1 p-2 border rounded w-full"
            value={selectedDistrict}
            onChange={handleDistrictChange}
            disabled={!districts.length}
          >
            <option value="">-- Choose District --</option>
            {districts.map((district) => (
              <option key={district} value={district}>
                {district}
              </option>
            ))}
          </select>
        </>
      )}
    </div>
  );
};

export default LocationSelector;
