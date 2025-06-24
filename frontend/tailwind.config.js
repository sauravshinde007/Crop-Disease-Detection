/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class', // or 'media' if you want automatic OS detection
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}