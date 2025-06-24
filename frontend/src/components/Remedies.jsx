import React from "react";

const RemediesDisplay = ({ remedies }) => {
  return (
    remedies && (
      <div className="mt-6 p-6 bg-green-100 shadow-lg rounded-lg w-full">
        <h2 className="text-xl font-semibold mb-3 text-gray-800">🩺 Remedies:</h2>
        <ul className="list-disc pl-6 text-left text-gray-700">
          {remedies.split(".").map((remedy, index) =>
            remedy.trim() ? (
              <li key={index} className="mb-2">
                {remedy.includes(":") ? (
                  <span className="font-bold text-green-700">{remedy}</span>
                ) : (
                  <span>{remedy}</span>
                )}
              </li>
            ) : null
          )}
        </ul>
      </div>
    )
  );
};

export default RemediesDisplay;
