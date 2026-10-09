import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/manrope'
import '@fontsource-variable/space-grotesk'
import './index.css'
import { installAuthFetch } from './auth.js'
import AppErrorBoundary from './components/AppErrorBoundary.jsx'
import App from './App.jsx'

installAuthFetch()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>,
)
