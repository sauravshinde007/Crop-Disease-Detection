import React from "react";
import RemediesDisplay from "./Remedies";

const ResultDisplay = ({ result }) => {
  return (
    result && (
      <div className="mt-6 p-6 bg-gray-100 shadow-lg rounded-lg w-full text-left">
        <h2 className="text-xl font-semibold mb-3 text-gray-800">📊 Detection Results:</h2>

        {result.confidence * 100 >= 50 ? (
          <>
            <p className="text-green-600 font-bold text-lg">🌱 Crop: {result.crop}</p>
            <p className="text-red-500 font-bold text-lg">🦠 Disease: {result.disease}</p>
            <p className="text-blue-600 font-bold text-lg">
              🔍 Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>

            {/* Weather Data */}
            
            {result.weather && (
              <div className="mt-4 p-4 bg-blue-100 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800">🌤 Weather Details</h3>
                
                <p>🌡 Temperature: {result.weather.temperature}°C</p>
                <p>💧 Humidity: {result.weather.humidity}%</p>
                <p>⛅ Condition: {result.weather.condition}</p>
              </div>
            )}

            {/* Remedies */}
            <RemediesDisplay remedies={result.remedy} />
          </>
        ) : (
          <p className="text-red-500 font-bold text-xl">⚠️ Unable to predict.</p>
        )}
      </div>
    )
  );
};

export default ResultDisplay;
