/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#0a0e14',
          secondary: '#111920',
          card: '#151d27'
        },
        accent: {
          cyan: '#00d4ff',
          green: '#00ff88',
          red: '#ff3366',
          yellow: '#ffd700'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace']
      }
    }
  },
  plugins: []
}
