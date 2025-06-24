import React from "react";

const Footer = () => {
  return (
    <footer id="footer" className="bg-green-800 text-white py-6 mt-12">
      <div className="max-w-6xl mx-auto px-4 text-center">
        <p>&copy; {new Date().getFullYear()} Pisheti. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
