/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        t1: "var(--t1)",
        t2: "var(--t2)",
        t3: "var(--t3)",
        nav: "var(--nav)",
        "nav-hover": "var(--nav-hover)",
        signal: "var(--signal)",
        "signal-light": "var(--signal-light)",
        "signal-border": "var(--signal-border)",
        boundary: "var(--boundary)",
        "boundary-light": "var(--boundary-light)",
        success: "var(--success)",
        "code-bg": "var(--code-bg)",
        "code-text": "var(--code-text)",
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
      },
      letterSpacing: {
        heading: "-0.3px",
      }
    },
  },
  plugins: [],
}
