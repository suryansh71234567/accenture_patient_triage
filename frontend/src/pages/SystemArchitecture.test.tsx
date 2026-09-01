import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SystemArchitecture } from "./SystemArchitecture";

describe("SystemArchitecture", () => {
  it("renders the static pipeline with no API calls", () => {
    render(<SystemArchitecture />);
    expect(screen.getByText("System Architecture")).toBeInTheDocument();
    expect(screen.getByText("PATIENT")).toBeInTheDocument();
    expect(screen.getByText("FINAL PATIENT FLOW")).toBeInTheDocument();
    expect(screen.getByText("Multi-Hospital Isolation")).toBeInTheDocument();
  });
});
