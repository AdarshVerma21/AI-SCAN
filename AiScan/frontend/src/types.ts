export type PredictionResponse = {
  prediction: "AI-generated" | "Real" | string | null;
  ai_score: number | null;
  threshold: number | null;
  model_scores: Record<string, number | null>;
  processing_time_ms: number;
  status: "success" | "model_not_ready" | "inference_error" | string;
  warnings: string[];
  error?: string | null;
  uncertainty: boolean;
  image?: {
    format?: string;
    width?: number;
    height?: number;
  } | null;
};

export type ApiError = {
  detail?: string;
};

