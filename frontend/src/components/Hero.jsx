import React from "react";
import heroImage from "../assets/pexels-jplenio-1574547.jpg";

const Hero = () => {
  return (
    <section id="hero" className="relative h-[60vh] flex items-center justify-center text-center text-white bg-black/70 bg-blend-overlay" style={{ backgroundImage: `url(${heroImage})`, backgroundSize: 'cover', backgroundPosition: 'center' }}>
      <div className="px-4 max-w-3xl">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">Detect Crop Diseases Instantly</h1>
        <p className="text-lg md:text-xl">Empowering farmers with AI-driven solutions to protect crops and improve yield.</p>
      </div>
    </section>
  );
};

export default Hero;
