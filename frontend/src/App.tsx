import { BrowserRouter, Route, Routes } from "react-router-dom";
import { SessionProvider } from "./state/SessionContext";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { LiveHospital } from "./pages/LiveHospital";
import { PatientList } from "./pages/PatientList";
import { PatientWorkspace } from "./pages/PatientWorkspace";

export default function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="live" element={<LiveHospital />} />
            <Route path="patients" element={<PatientList />} />
            <Route path="patients/:id" element={<PatientWorkspace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  );
}
