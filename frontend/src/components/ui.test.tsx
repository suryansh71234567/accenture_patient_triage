import { describe, expect, it } from "vitest";
import { acuityBorderColor, acuityLabel, acuityTone, deptStatusTone, fmtWaitMinutes } from "./ui";

describe("acuityTone / acuityLabel", () => {
  it("maps acuity 1 to critical/Critical", () => {
    expect(acuityTone(1)).toBe("critical");
    expect(acuityLabel(1)).toBe("Critical");
  });
  it("maps null acuity to neutral/Unassessed", () => {
    expect(acuityTone(null)).toBe("neutral");
    expect(acuityLabel(null)).toBe("Unassessed");
  });
  it("maps acuity 5 to good/Minor", () => {
    expect(acuityTone(5)).toBe("good");
    expect(acuityLabel(5)).toBe("Minor");
  });
});

describe("acuityBorderColor", () => {
  it("returns a css var per tone", () => {
    expect(acuityBorderColor("critical")).toBe("var(--color-critical-500)");
    expect(acuityBorderColor("neutral")).toBe("var(--color-border)");
  });
});

describe("deptStatusTone", () => {
  it("reports Available under 80% occupancy", () => {
    const s = deptStatusTone(5, 10, "OPEN");
    expect(s.tone).toBe("good");
    expect(s.label).toBe("Available");
    expect(s.pct).toBe(50);
  });
  it("reports High load at 80-94%", () => {
    const s = deptStatusTone(8, 10, "OPEN");
    expect(s.tone).toBe("warn");
    expect(s.label).toBe("High load");
  });
  it("reports Full at 95%+", () => {
    const s = deptStatusTone(10, 10, "OPEN");
    expect(s.tone).toBe("critical");
    expect(s.label).toBe("Full");
  });
  it("reports Closed status regardless of occupancy", () => {
    const s = deptStatusTone(0, 10, "CLOSED");
    expect(s.closed).toBe(true);
    expect(s.label).toBe("Closed");
    expect(s.pct).toBe(100);
  });
  it("handles zero capacity without dividing by zero", () => {
    const s = deptStatusTone(0, 0, "OPEN");
    expect(s.pct).toBe(0);
  });
});

describe("fmtWaitMinutes", () => {
  it("formats under an hour as Nm", () => {
    expect(fmtWaitMinutes(0)).toBe("0m");
    expect(fmtWaitMinutes(45)).toBe("45m");
  });
  it("formats an hour or more as NhMm", () => {
    expect(fmtWaitMinutes(60)).toBe("1h 0m");
    expect(fmtWaitMinutes(125)).toBe("2h 5m");
  });
});
