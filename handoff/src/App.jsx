import LandingV4Final from './components/LandingV4Final.jsx';
import AppPage from './components/AppPage.jsx';

const path = window.location.pathname;
const isApp = path === '/app' || path.startsWith('/app/');

export default function App() {
  return isApp ? <AppPage /> : <LandingV4Final />;
}
