import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Services from "./components/Services";
import Footer from "./components/Footer";
import LocationSelector from "./components/LocationSelector";
import CropSelector from "./components/CropSelector";
import ImageUploader from "./components/ImageUploader";
import DetectButton from "./components/DetectButton";
import ResultDisplay from "./components/ResultDisplay";
import LanguageDropdown from "./components/DropDown";
import bgImage from "./assets/pexels-jplenio-1574547.jpg";


const App = () => {
  const [selectedCrop, setSelectedCrop] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState({ state: "", district: "" });
  const [selectedLanguage, setSelectedLanguage] = useState("en");

  const handleLocationSelect = (state, district) => {
    setLocation({ state, district });
  };

  const handleDetectDisease = async () => {
    if (!selectedCrop || !image || !location.state || !location.district) {
      alert("Please select a crop, upload an image, and choose location.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);
    formData.append("crop", selectedCrop);
    formData.append("state", location.state);
    formData.append("district", location.district);
    formData.append("language", selectedLanguage);

    try {
      const response = await fetch("http://localhost:8000/api/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Detection Error:", error);
      alert("Error processing image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-fixed bg-cover" style={{ backgroundImage: `url(${bgImage})` }}>
      <Navbar />
      <Hero />
      <Services />

      <section
  id="detection"
  className="relative z-10 bg-white/70 backdrop-blur-lg rounded-3xl shadow-2xl p-6 md:p-10 w-full max-w-lg mx-auto mt-16 mb-16 border border-white/30 transition-all duration-300"
>
  <div className="flex flex-col gap-y-6">
    <LanguageDropdown
      selectedLanguage={selectedLanguage}
      onLanguageChange={setSelectedLanguage}
    />

    <LocationSelector onLocationSelect={handleLocationSelect} />

    {location.state && location.district && (
      <div className="p-3 bg-blue-50/80 shadow rounded-lg text-center w-full">
        <h2 className="text-base font-semibold text-blue-900">📍 Selected Location</h2>
        <p className="text-blue-700 font-bold">{location.state}, {location.district}</p>
      </div>
    )}

    <CropSelector selectedCrop={selectedCrop} setSelectedCrop={setSelectedCrop} />
    <ImageUploader setImage={setImage} setPreview={setPreview} preview={preview} />
    <DetectButton onClick={handleDetectDisease} loading={loading} />
    <ResultDisplay result={result} />
  </div>
</section>

      <Footer />
    </div>
  );
};

export default App;
