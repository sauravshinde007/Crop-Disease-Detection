import React, { useEffect, useState } from "react";

const GetWeather = ({ state, district, setWeather }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (state && district) {
      fetchWeather();
    }
  }, [state, district]);

  const fetchWeather = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/api/weather/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, district }),
      });

      const data = await response.json();
      if (response.ok) {
        setWeather(data);
      } else {
        setError(data.error || "Failed to fetch weather.");
      }
    } catch (err) {
      setError("Error fetching weather data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-blue-100 rounded-md shadow-md">
      <h2 className="text-lg font-semibold text-blue-800">🌤 Weather Details</h2>
      {loading && <p>Loading weather data...</p>}
      {error && <p className="text-red-600">{error}</p>}
    </div>
  );
};

export default GetWeather;
