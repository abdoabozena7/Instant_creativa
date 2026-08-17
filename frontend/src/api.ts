import type { AnswerResponse, HealthResponse, MetricsResponse, Mode, SearchResponse, VisionAnswerResponse } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  metrics: () => request<MetricsResponse>("/api/metrics"),
  search: (query: string, mode: Mode, site: string) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        mode,
        top_k: 8,
        cancer_sites: site ? [site] : [],
      }),
    }),
  answer: (query: string, mode: Mode, site: string) =>
    request<AnswerResponse>("/api/answer", {
      method: "POST",
      body: JSON.stringify({
        query,
        mode,
        evidence_k: 6,
        cancer_sites: site ? [site] : [],
      }),
    }),
  visionAnswer: (imageBase64: string, mimeType: string, caseContext: string, mode: Mode, site: string) =>
    request<VisionAnswerResponse>("/api/vision/answer", {
      method: "POST",
      body: JSON.stringify({
        image_base64: imageBase64,
        mime_type: mimeType,
        case_context: caseContext,
        mode,
        cancer_sites: site ? [site] : [],
        privacy_confirmed: true,
      }),
    }),
};
