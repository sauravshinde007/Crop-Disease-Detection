import React from "react";

const DetectButton = ({ onClick, loading }) => {
  return (
    <button
      onClick={onClick}
      className="w-full py-3 mt-4 text-white bg-green-600 rounded-lg shadow-md hover:bg-green-800 transition duration-300 "
      disabled={loading}
    >
      {loading ? "Processing..." : "Detect Disease"}
    </button>
  );
};

export default DetectButton;
