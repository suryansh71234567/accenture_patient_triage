import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useSession } from "../state/SessionContext";
import type { HospitalInfo } from "../types";

/** Small dropdown letting the nurse pick which registered hospital the app is scoped to. */
export function HospitalSelector() {
  const { hospitalId, setHospitalId, mutationTick } = useSession();
  const [hospitals, setHospitals] = useState<HospitalInfo[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .listHospitals()
      .then((list) => {
        if (!cancelled) setHospitals(list);
      })
      .catch(() => {
        // Non-fatal — selector just falls back to showing the current hospitalId alone.
      });
    return () => {
      cancelled = true;
    };
    // Refetch on mutationTick so a newly registered hospital (via the
    // onboarding page, a plain REST call outside proposeAction/sendChat)
    // shows up here without a full page reload.
  }, [mutationTick]);

  const knownIds = new Set(hospitals.map((h) => h.hospital_id));

  return (
    <select
      value={hospitalId}
      onChange={(e) => setHospitalId(e.target.value)}
      title="Active hospital"
      className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-brand-400)]"
    >
      {!knownIds.has(hospitalId) && <option value={hospitalId}>{hospitalId}</option>}
      {hospitals.map((h) => (
        <option key={h.hospital_id} value={h.hospital_id}>
          {h.hospital_name}
        </option>
      ))}
    </select>
  );
}
