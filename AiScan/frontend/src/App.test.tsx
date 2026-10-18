import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function imageFile(name = "sample.png", type = "image/png") {
  return new File(["valid-image"], name, { type });
}

describe("AiScan dashboard", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", { ...URL, createObjectURL: vi.fn(() => "blob:preview"), revokeObjectURL: vi.fn() });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("validates unsupported files before calling the backend", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [imageFile("sample.gif", "image/gif")] } });
    expect(screen.getByRole("alert")).toHaveTextContent("Choose a JPEG, PNG, or WebP image.");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows a real backend result without inventing values", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        prediction: "Real",
        ai_score: 0.18,
        threshold: 0.63,
        model_scores: { SAFE: -1.2, "PROBE-DINOv2": -0.8 },
        processing_time_ms: 112,
        status: "success",
        warnings: [],
        uncertainty: false,
      }),
    }));
    render(<App />);
    fireEvent.change(document.querySelector('input[type="file"]') as HTMLInputElement, { target: { files: [imageFile()] } });
    fireEvent.click(screen.getByRole("button", { name: /^Analyze image$/i }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Real" })).toBeInTheDocument());
    expect(screen.getByText("18%")).toBeInTheDocument();
    expect(screen.getByText("-1.200")).toBeInTheDocument();
  });

  it("shows model-not-ready responses as setup errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        prediction: null,
        ai_score: null,
        threshold: null,
        model_scores: {},
        processing_time_ms: 8,
        status: "model_not_ready",
        warnings: ["fusion artifact is unavailable"],
        uncertainty: false,
      }),
    }));
    render(<App />);
    fireEvent.change(document.querySelector('input[type="file"]') as HTMLInputElement, { target: { files: [imageFile()] } });
    fireEvent.click(screen.getByRole("button", { name: /^Analyze image$/i }));
    await waitFor(() => expect(screen.getByText("Models aren’t ready yet")).toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent("fusion artifact is unavailable");
  });
});
