import { getAuthToken } from '@/services/auth';
import { getEnvConfig } from '@/config/env';

const { useMockApi } = getEnvConfig();

/** Base prefix for all backend API routes */
const API_PREFIX = '/api/federated';

/* -------------------------------------------------------------------------- */
/*                                Types                                       */
/* -------------------------------------------------------------------------- */

export interface FederatedStatus {
  active: boolean;
  message?: string;
  registered?: boolean;
  current_round?: number;
  [key: string]: any;
}

export interface FederatedActionResponse {
  success: boolean;
  message: string;
  [key: string]: any;
}

/* -------------------------------------------------------------------------- */
/*                               Utilities                                    */
/* -------------------------------------------------------------------------- */

/** Construct headers, automatically attaching the auth token when present */
const getHeaders = (contentType?: string): HeadersInit => {
  const headers: Record<string, string> = {};

  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  if (contentType) {
    headers['Content-Type'] = contentType;
  }

  return headers;
};

/** Generic helper to check the fetch response */
const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    let errorMessage = `API error: ${response.status}`;
    try {
      const data = await response.json();
      errorMessage = data?.message || errorMessage;
    } catch { /* ignore json parse errors */ }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
};

/** Simulate an API call when operating in mock mode */
const simulateApiCall = async <T>(mockData: T, delayMs = 500): Promise<T> => {
  return new Promise<T>((resolve) => setTimeout(() => resolve(mockData), delayMs));
};

/* -------------------------------------------------------------------------- */
/*                             API Functions                                  */
/* -------------------------------------------------------------------------- */

/**
 * Get the current federated-learning status from the backend.
 */
export const getFederatedStatus = async (): Promise<FederatedStatus> => {
  if (useMockApi) {
    return simulateApiCall<FederatedStatus>({
      active: false,
      message: 'Mock – federated learning disabled',
    });
  }

  const res = await fetch(`${API_PREFIX}/status`, {
    headers: getHeaders(),
  });
  return handleResponse<FederatedStatus>(res);
};

/**
 * Manually trigger participation in the current federated-learning round.
 */
export const triggerFederatedRound = async (): Promise<FederatedActionResponse> => {
  if (useMockApi) {
    return simulateApiCall<FederatedActionResponse>({
      success: true,
      message: 'Mock – triggered federated round',
    });
  }

  const res = await fetch(`${API_PREFIX}/trigger-round`, {
    method: 'POST',
    headers: getHeaders('application/json'),
  });
  return handleResponse<FederatedActionResponse>(res);
};

/**
 * Request the latest global model from the coordinator and apply it on the node.
 */
export const syncFederatedModel = async (): Promise<FederatedActionResponse> => {
  if (useMockApi) {
    return simulateApiCall<FederatedActionResponse>({
      success: true,
      message: 'Mock – synced global model',
    });
  }

  const res = await fetch(`${API_PREFIX}/sync-model`, {
    method: 'POST',
    headers: getHeaders('application/json'),
  });
  return handleResponse<FederatedActionResponse>(res);
}; 