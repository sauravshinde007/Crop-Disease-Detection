import React from "react";

const Services = () => {
  const services = [
    {
      title: "AI-Based Detection",
      desc: "Upload crop images and let AI detect diseases with high accuracy.",
    },
    {
      title: "Localized Remedies",
      desc: "Get location-based solutions and weather-aware treatments.",
    },
    {
      title: "Multilingual Support",
      desc: "Use the app in your preferred language for ease of access.",
    },
  ];

  return (
    <section id="services" className="py-16 bg-white">
      <div className="max-w-6xl mx-auto px-4 text-center">
        <h2 className="text-3xl font-bold mb-8 text-green-700">Our Services</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {services.map((service, idx) => (
            <div key={idx} className="p-6 bg-green-50 rounded-xl shadow hover:shadow-lg transition">
              <h3 className="text-xl font-semibold text-green-800 mb-2">{service.title}</h3>
              <p className="text-gray-700">{service.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Services;
