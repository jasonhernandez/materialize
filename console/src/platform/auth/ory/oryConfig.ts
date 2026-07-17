/**
 * Ory Elements configuration for the Console.
 *
 * This module provides the Ory client configuration used by all Ory Elements
 * flow components (Login, Registration, Settings, etc.)
 */

import { appConfig } from "~/config/AppConfig";
import { getOryProjectUrl } from "~/config/oryUrls";
import {
  FetchConfiguration,
  FetchFrontendApi,
} from "~/external-library-wrappers/ory";
import type { OryClientConfiguration } from "~/external-library-wrappers/ory-elements";

/**
 * Get the current Ory project base URL.
 */
function getOryBaseUrl(): string {
  const stack =
    appConfig.mode === "cloud"
      ? (appConfig.currentStack ?? "staging")
      : "staging";
  return getOryProjectUrl(stack);
}

/**
 * Create an Ory FrontendApi client configured for the current environment.
 */
export function createOryFrontendApi(): FetchFrontendApi {
  const config = new FetchConfiguration({
    basePath: getOryBaseUrl(),
    credentials: "include", // Include cookies for session management
    // Without an explicit JSON accept, Kratos treats self-service flow
    // requests as page loads and 303s to the hosted UI — which an SPA's
    // cross-origin fetch cannot follow (CORS). We always want flow JSON.
    headers: { Accept: "application/json" },
  });
  return new FetchFrontendApi(config);
}

/**
 * Build the Ory client configuration for use with Ory Elements components.
 *
 * @param options - Optional configuration overrides
 */
export function buildOryClientConfig(options?: {
  locale?: string;
}): OryClientConfiguration {
  // Note: The project config is simplified here. The full AccountExperienceConfiguration
  // type has many optional fields that are configured in the Ory Console, not in code.
  // We provide minimal config and let Ory Elements handle defaults.
  return {
    sdk: {
      // The SDK URL for API calls
      url: getOryBaseUrl(),
    },
    project: {
      // Project name for branding
      name: "Materialize",
    } as OryClientConfiguration["project"],
    intl: {
      locale: options?.locale ?? "en",
    },
  };
}

/**
 * Flow URL builders for initiating Ory self-service flows.
 * These URLs redirect to the Ory-hosted flow initialization endpoints.
 */
export const oryFlowUrls = {
  /**
   * Get the URL to initiate a login flow.
   * @param returnTo - Optional URL to redirect to after successful login
   */
  login: (returnTo?: string) => {
    const baseUrl = getOryBaseUrl();
    const url = new URL(`${baseUrl}/self-service/login/browser`);
    if (returnTo) {
      url.searchParams.set("return_to", returnTo);
    }
    return url.toString();
  },

  /**
   * Get the URL to initiate a registration flow.
   * @param returnTo - Optional URL to redirect to after successful registration
   */
  registration: (returnTo?: string) => {
    const baseUrl = getOryBaseUrl();
    const url = new URL(`${baseUrl}/self-service/registration/browser`);
    if (returnTo) {
      url.searchParams.set("return_to", returnTo);
    }
    return url.toString();
  },

  /**
   * Get the URL to initiate a recovery flow (forgot password).
   */
  recovery: () => {
    const baseUrl = getOryBaseUrl();
    return `${baseUrl}/self-service/recovery/browser`;
  },

  /**
   * Get the URL to initiate a verification flow (verify email).
   */
  verification: () => {
    const baseUrl = getOryBaseUrl();
    return `${baseUrl}/self-service/verification/browser`;
  },

  /**
   * Get the URL to initiate a settings flow (account settings).
   */
  settings: () => {
    const baseUrl = getOryBaseUrl();
    return `${baseUrl}/self-service/settings/browser`;
  },

  /**
   * Get the URL to initiate a logout flow.
   */
  logout: () => {
    const baseUrl = getOryBaseUrl();
    return `${baseUrl}/self-service/logout/browser`;
  },
};
