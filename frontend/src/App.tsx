import { BrowserRouter, Route, Routes } from "react-router-dom";
import { SessionProvider } from "./state/SessionContext";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { LiveHospital } from "./pages/LiveHospital";
import { HospitalNetwork } from "./pages/HospitalNetwork";
import { PatientList } from "./pages/PatientList";
import { PatientWorkspace } from "./pages/PatientWorkspace";
import { ClinicalIntelligence } from "./pages/ClinicalIntelligence";
import { SimulationControlCenter } from "./pages/SimulationControlCenter";
import { SystemArchitecture } from "./pages/SystemArchitecture";

export default function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="live" element={<LiveHospital />} />
            <Route path="network" element={<HospitalNetwork />} />
            <Route path="patients" element={<PatientList />} />
            <Route path="patients/:id" element={<PatientWorkspace />} />
            <Route path="intelligence" element={<ClinicalIntelligence />} />
            <Route path="simulation" element={<SimulationControlCenter />} />
            <Route path="architecture" element={<SystemArchitecture />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  );
}
