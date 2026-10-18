import type { ApiError, PredictionResponse } from "./types";

export const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function predictImage(file: File, signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: form,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    throw new Error("The backend is unavailable. Check that the API is running.");
  }

  const payload = (await response.json().catch(() => null)) as PredictionResponse | ApiError | null;
  if (!response.ok) {
    const detail = payload && "detail" in payload ? payload.detail : undefined;
    throw new Error(detail || `The backend returned HTTP ${response.status}.`);
  }
  return payload as PredictionResponse;
}

