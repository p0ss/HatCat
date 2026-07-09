// Common envelope, error shape, and long-running-operation handles.
// Cross-slice contract — locked in P0.2.

export type ApiMeta = {
  fetched_at: string; // ISO 8601
  source?: string;
};

export type ApiResponse<T> = {
  data: T;
  _meta: ApiMeta;
};

export type ApiErrorBody = {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(
    code: string,
    message: string,
    status: number,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

// Long-running operations return a job handle immediately.
// Status is then polled (or streamed) via job-specific endpoints.
export type JobHandle = {
  job_id: string;
  resource_type: "run" | "meld";
  resource_id: string;
  poll_url: string;
  stream_url?: string;
};

// Everything that wraps a paginated list (when not using search).
export type Page<T> = {
  items: T[];
  total: number;
  next_cursor?: string;
};
