import logo from "../assets/pisheti_logo.png"; // Update path if needed

const Navbar = () => {
  return (
    <nav className="bg-white/90 backdrop-blur sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <img
            src={logo}
            alt="Logo"
            className="h-10 w-auto rounded bg-white p-1 shadow"
          />
        </div>

        {/* Navigation Links */}
        <ul className="flex gap-6 text-gray-700 font-medium">
          <li><a href="#hero" className="hover:text-green-600">Home</a></li>
          <li><a href="#services" className="hover:text-green-600">Services</a></li>
          <li><a href="#detection" className="hover:text-green-600">Detection</a></li>
          <li><a href="#footer" className="hover:text-green-600">Contact</a></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
