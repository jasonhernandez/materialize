/**
 * Auth Provider Detection Module
 *
 * This module determines which authentication provider (Frontegg or Ory) to use
 * BEFORE authentication occurs. This solves the chicken-and-egg problem where
 * feature flags require authentication, but authentication requires knowing
 * which provider to use.
 *
 * Detection happens in priority order:
 * 1. URL parameter (developer testing)
 * 2. Existing session token (returning users)
 * 3. localStorage preference (sticky routing)
 * 4. Email-first discovery via the sync-server (new user routing)
 * 5. Default to Frontegg (fallback)
 */

import { appConfig } from "~/config/AppConfig";

export type AuthProviderType = "frontegg" | "ory";

const STORAGE_KEY = "mz-auth-provider";
const LAST_EMAIL_KEY = "mz-last-login-email";
const URL_PARAM = "auth_provider";
const EMAIL_URL_PARAM = "email";

// Timeout for the discovery call to prevent blocking in CI/slow networks
const DISCOVERY_TIMEOUT_MS = 2000;

/**
 * Detect which auth provider to use.
 * Called BEFORE authentication, so no user context is available.
 *
 * Detection is ordered to minimize latency:
 * - Synchronous checks first (URL param, existing session, localStorage)
 * - Async discovery call only if needed (with timeout)
 */
export async function detectAuthProvider(): Promise<AuthProviderType> {
  // 1. Check URL parameter first (for testing/development/E2E)
  // This allows E2E tests to explicitly force a provider without any async calls
  const urlProvider = detectFromUrlParam();
  if (urlProvider) {
    persistProviderChoice(urlProvider);
    return urlProvider;
  }

  // 2. Check for existing token and determine its issuer
  const existingProvider = detectFromExistingSession();
  if (existingProvider) {
    return existingProvider;
  }

  // 3. Check localStorage for sticky preference
  const storedProvider = detectFromStorage();
  if (storedProvider) {
    return storedProvider;
  }

  // 4. Email-first discovery via the sync-server. LaunchDarkly evaluation
  // (the `ory-auth-enabled` flag, keyed by email domain) happens
  // server-side; the browser never talks to LD pre-auth.
  const discoveredProvider = await detectFromDiscovery();
  if (discoveredProvider) {
    return discoveredProvider;
  }

  // 5. Default to Frontegg
  return "frontegg";
}

/**
 * Helper to add a timeout to a promise.
 * Returns the default value if the promise doesn't resolve in time.
 */
async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  defaultValue: T,
): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout>;
  const timeoutPromise = new Promise<T>((resolve) => {
    timeoutId = setTimeout(() => resolve(defaultValue), timeoutMs);
  });

  try {
    const result = await Promise.race([promise, timeoutPromise]);
    clearTimeout(timeoutId!);
    return result;
  } catch {
    clearTimeout(timeoutId!);
    return defaultValue;
  }
}

/**
 * Email-first provider discovery via the sync-server
 * (`POST /api/auth/discovery`, see the Ory migration design doc). The
 * backend evaluates the `ory-auth-enabled` LaunchDarkly flag against the
 * email's domain and defaults to Frontegg on any failure.
 *
 * Discovery needs an email, which is only available pre-auth when a login
 * UI has captured one (persisted under `mz-last-login-email`) or an
 * `?email=` param is present (testing / deep links). Without a hint this
 * step is skipped.
 */
async function detectFromDiscovery(): Promise<AuthProviderType | null> {
  if (appConfig.mode !== "cloud" || !("cloudGlobalApiUrl" in appConfig)) {
    return null;
  }

  const email = getEmailHint();
  if (!email) {
    return null;
  }

  return withTimeout(
    (async () => {
      try {
        const response = await fetch(
          `${appConfig.cloudGlobalApiUrl}/api/auth/discovery`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ email }),
          },
        );
        if (!response.ok) {
          return null;
        }
        const body: { provider?: string } = await response.json();
        return body.provider === "ory" || body.provider === "frontegg"
          ? body.provider
          : null;
      } catch {
        // Discovery failed; fall through to the Frontegg default
        return null;
      }
    })(),
    DISCOVERY_TIMEOUT_MS,
    null, // Default to null (use Frontegg) on timeout
  );
}

/**
 * The email to run discovery against, when one is known pre-auth.
 */
function getEmailHint(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get(EMAIL_URL_PARAM);
    if (fromUrl && fromUrl.includes("@")) {
      return fromUrl;
    }
  } catch {
    // window.location may not be available in some contexts
  }
  try {
    const stored = localStorage.getItem(LAST_EMAIL_KEY);
    if (stored && stored.includes("@")) {
      return stored;
    }
  } catch {
    // localStorage may not be available
  }
  return null;
}

/**
 * Remember the email a user logged in (or attempted to) with, so future
 * visits can run email-first discovery before any provider UI loads.
 */
export function persistLoginEmail(email: string): void {
  try {
    localStorage.setItem(LAST_EMAIL_KEY, email);
  } catch {
    // localStorage may not be available
  }
}

/**
 * Detect provider from existing session tokens.
 * Checks both Frontegg and Ory token storage locations.
 */
function detectFromExistingSession(): AuthProviderType | null {
  // Check Frontegg token in localStorage
  // Frontegg stores tokens with various key patterns
  try {
    // Check for Frontegg access token
    const fronteggKeys = Object.keys(localStorage).filter(
      (key) =>
        key.includes("frontegg") ||
        key.includes("fe_") ||
        key.startsWith("FE_"),
    );
    for (const key of fronteggKeys) {
      const value = localStorage.getItem(key);
      if (value && looksLikeJwt(value) && !isTokenExpired(value)) {
        return "frontegg";
      }
    }
  } catch {
    // localStorage may not be available
  }

  // Check Ory token in sessionStorage
  try {
    const oryToken = sessionStorage.getItem("ory_access_token");
    if (oryToken && !isTokenExpired(oryToken)) {
      return "ory";
    }
  } catch {
    // sessionStorage may not be available
  }

  return null;
}

/**
 * Detect provider from URL parameter.
 * Used for developer testing: ?auth_provider=ory
 */
function detectFromUrlParam(): AuthProviderType | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const param = params.get(URL_PARAM);
    if (param === "ory" || param === "frontegg") {
      return param;
    }
  } catch {
    // window.location may not be available in some contexts
  }
  return null;
}

/**
 * Detect provider from localStorage sticky preference.
 * Once a user authenticates with a provider, we remember it.
 */
function detectFromStorage(): AuthProviderType | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "ory" || stored === "frontegg") {
      return stored;
    }
  } catch {
    // localStorage may not be available
  }
  return null;
}

/**
 * Whether an Ory session is currently active.
 *
 * Used at API-call time to pick which provider's token to attach: the
 * module-level apiClient singleton is constructed before async provider
 * detection completes (and before the OAuth callback stores the Ory
 * token), so this must be checked per call rather than at client
 * construction.
 */
export function isOryProviderActive(): boolean {
  try {
    const oryToken = sessionStorage.getItem("ory_access_token");
    return !!oryToken && !isTokenExpired(oryToken);
  } catch {
    return false;
  }
}

/**
 * Persist the provider choice for returning users.
 * Called after successful authentication or when URL param is used.
 */
export function persistProviderChoice(provider: AuthProviderType): void {
  try {
    localStorage.setItem(STORAGE_KEY, provider);
  } catch {
    // localStorage may not be available
  }
}

/**
 * Clear the stored provider choice.
 * Called during logout or emergency rollback.
 */
export function clearProviderChoice(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // localStorage may not be available
  }
}

/**
 * Check if a string looks like a JWT (has 3 dot-separated parts).
 */
function looksLikeJwt(value: string): boolean {
  const parts = value.split(".");
  return parts.length === 3;
}

/**
 * Check if a JWT token is expired.
 */
function isTokenExpired(token: string): boolean {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return true;

    const payload = JSON.parse(atob(parts[1]));
    if (!payload.exp) return false; // No expiration claim

    // Add 30 second buffer to account for clock skew
    return Date.now() >= payload.exp * 1000 - 30000;
  } catch {
    return true;
  }
}
