/**
 * Ory OIDC Configuration
 *
 * Configures oidc-client-ts UserManager for OAuth2 Authorization Code flow
 * with PKCE against Ory Network.
 */

import { UserManager, WebStorageStateStore } from "oidc-client-ts";

import { appConfig } from "~/config/AppConfig";

/**
 * Create a UserManager instance configured for Ory OAuth2.
 *
 * The UserManager handles:
 * - OAuth2 Authorization Code flow with PKCE
 * - Token storage in sessionStorage
 * - State/nonce management for CSRF protection
 * - Automatic token refresh (when configured)
 *
 * Note: We use explicit metadata instead of discovery to avoid CORS issues
 * when the Ory project doesn't have localhost in its allowed origins.
 */
export function createOryUserManager(): UserManager {
  if (appConfig.mode !== "cloud") {
    throw new Error("Ory UserManager is only available in cloud mode");
  }

  const { oryProjectUrl, oryOAuth2ClientId } = appConfig;

  if (!oryProjectUrl || !oryOAuth2ClientId) {
    throw new Error(
      "Ory configuration missing. Set VITE_ORY_PROJECT_URL and VITE_ORY_OAUTH2_CLIENT_ID.",
    );
  }

  return new UserManager({
    // Authority is required even with explicit metadata
    authority: oryProjectUrl,

    // OAuth2 client ID (public client, no secret)
    client_id: oryOAuth2ClientId,

    // Callback URL after Ory login completes
    // Must match what's registered in the Ory OAuth2 client
    redirect_uri: `${window.location.origin}/callback`,

    // Post-logout redirect
    post_logout_redirect_uri: window.location.origin,

    // Authorization Code flow (PKCE is automatic for public clients)
    response_type: "code",

    // Scopes to request
    scope: "openid email profile",

    // Store tokens in sessionStorage (cleared on tab close)
    userStore: new WebStorageStateStore({ store: sessionStorage }),

    // Disable automatic token refresh for POC
    automaticSilentRenew: false,

    // Skip userinfo endpoint (we get claims from id_token)
    loadUserInfo: false,

    // Skip metadata fetch to avoid CORS issues
    // oidc-client-ts will use the explicit metadata below instead of fetching
    metadataUrl: undefined,

    // Explicit metadata to avoid CORS issues with /.well-known/openid-configuration
    metadata: {
      issuer: oryProjectUrl,
      authorization_endpoint: `${oryProjectUrl}/oauth2/auth`,
      token_endpoint: `${oryProjectUrl}/oauth2/token`,
      userinfo_endpoint: `${oryProjectUrl}/userinfo`,
      end_session_endpoint: `${oryProjectUrl}/oauth2/sessions/logout`,
      jwks_uri: `${oryProjectUrl}/.well-known/jwks.json`,
    },
  });
}

/**
 * Singleton UserManager instance.
 * Lazily initialized to avoid errors during SSR or non-cloud modes.
 */
let userManagerInstance: UserManager | null = null;

/**
 * Get or create the singleton UserManager instance.
 */
export function getOryUserManager(): UserManager {
  if (!userManagerInstance) {
    userManagerInstance = createOryUserManager();
  }
  return userManagerInstance;
}

/**
 * Clear the UserManager instance (useful for testing or logout).
 */
export function clearOryUserManager(): void {
  userManagerInstance = null;
}
