import React from "react";

const languages = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "ta", name: "Tamil" },
  { code: "bn", name: "Bengali" },
  { code: "mr", name: "Marathi" },
  { code: "kn", name: "Kannada" },
  { code: "gu", name: "Gujarati" },
  { code: "ml", name: "Malayalam" },
  { code: "te", name: "Telugu" },
];

const LanguageDropdown = ({ selectedLanguage, onLanguageChange }) => {
  return (
    <div className="language-dropdown">
      <label htmlFor="language">Select Language:</label>
      <select
        id="language"
        value={selectedLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
      >
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LanguageDropdown;
