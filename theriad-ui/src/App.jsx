import { lazy, Suspense, useEffect, useState, useTransition } from 'react'
import AppErrorBoundary from './components/AppErrorBoundary.jsx'
import { LoadingScreen } from './components/LoadingScreen.jsx'
import { useSiteAnalytics } from './siteAnalytics.js'
import { hasAuthenticatedSession } from './auth.js'

const loadLanding = () => import('./Landing.jsx')
const loadWorkspace = () => import('./Theriad.jsx')
const Landing = lazy(loadLanding)
const TheriadDashboard = lazy(loadWorkspace)

function getRoute() {
  const hash = window.location.hash
  const analytics = hash === '#/app/analytics'
  const isExplicitLanding = hash === '#/landing'
  const hasAuth = hasAuthenticatedSession()

  // An authenticated returning user visiting the root without a hash automatically
  // enters the workspace, while explicit #/landing or unauthenticated visitors see the landing page.
  const showApp = (hash === '#/app' || analytics) || (!isExplicitLanding && !hash && hasAuth)

  return {
    showApp,
    analytics,
  }
}

function App() {
  const [route, setRoute] = useState(getRoute)
  const [, startTransition] = useTransition()

  useSiteAnalytics()

  useEffect(() => {
    // If the returning authenticated user visited root, update the hash to reflect the workspace
    if (!window.location.hash && hasAuthenticatedSession()) {
      window.location.replace('#/app')
    }
  }, [])

  useEffect(() => {
    const handleHashChange = () => {
      startTransition(() => setRoute(getRoute()))
    }
    const handleUserChange = () => {
      startTransition(() => setRoute(getRoute()))
    }
    window.addEventListener('hashchange', handleHashChange)
    window.addEventListener('theriad:user-changed', handleUserChange)
    return () => {
      window.removeEventListener('hashchange', handleHashChange)
      window.removeEventListener('theriad:user-changed', handleUserChange)
    }
  }, [startTransition])

  useEffect(() => {
    // Warm the workspace chunk while a visitor reads the landing page. This
    // keeps the first click optimistic without making the initial bundle pay
    // for the full dashboard.
    if (!route.showApp) {
      const preload = window.setTimeout(() => { void loadWorkspace() }, 900)
      return () => window.clearTimeout(preload)
    }
    return undefined
  }, [route.showApp])

  const routeKey = route.showApp
    ? (route.analytics ? 'private-analytics' : 'app')
    : 'landing'

  return (
    <AppErrorBoundary key={routeKey} resetKey={routeKey}>
      <Suspense fallback={<LoadingScreen variant={route.showApp ? 'workspace' : 'landing'} />}>
        {route.showApp
          ? <TheriadDashboard key={routeKey} initialView={route.analytics ? 'Analytics' : 'Chat'} />
          : <Landing />}
      </Suspense>
    </AppErrorBoundary>
  )
}

export default App
