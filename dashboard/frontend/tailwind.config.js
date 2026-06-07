/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        nexus: {
          bg: "#0b0f14",
          panel: "#121820",
          accent: "#22d3ee",
          danger: "#f87171",
          muted: "#64748b",
        },
      },
    },
  },
  plugins: [],
};
