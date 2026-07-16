/**
 * Ory Token Management
 *
 * This module provides access to Ory OAuth2 access tokens for API authentication.
 * During the migration, this runs alongside fronteggToken.ts.
 *
 * TODO(ory-migration): Implement real token storage and retrieval once OAuth2 flow is ready
 */

import * as Sentry from "@sentry/react";

// Storage key for Ory access token in sessionStorage
const ORY_TOKEN_KEY = "ory_access_token";

/**
 * Returns the current Ory OAuth2 access token.
 *
 * In the real implementation, this will retrieve the token from sessionStorage
 * where it's stored after the OAuth2 PKCE flow completes.
 *
 * For now, this is a stub that returns null since Ory auth isn't implemented.
 */
export function getOryAccessToken(): string | null {
  try {
    const token = sessionStorage.getItem(ORY_TOKEN_KEY);

    if (!token) {
      Sentry.addBreadcrumb({
        level: "warning",
        category: "auth",
        message: "No Ory access token found",
      });
      return null;
    }

    return token;
  } catch {
    // sessionStorage may not be available
    Sentry.addBreadcrumb({
      level: "error",
      category: "auth",
      message: "Failed to access sessionStorage for Ory token",
    });
    return null;
  }
}

/**
 * Store the Ory access token after successful OAuth2 authentication.
 *
 * TODO(ory-migration): Called by OryProviderWrapper after OAuth2 flow completes
 */
export function setOryAccessToken(token: string): void {
  try {
    sessionStorage.setItem(ORY_TOKEN_KEY, token);
  } catch {
    Sentry.addBreadcrumb({
      level: "error",
      category: "auth",
      message: "Failed to store Ory access token in sessionStorage",
    });
  }
}

/**
 * Clear the Ory access token (called during logout).
 */
export function clearOryAccessToken(): void {
  try {
    sessionStorage.removeItem(ORY_TOKEN_KEY);
  } catch {
    // sessionStorage may not be available
  }
}

/**
 * Check if an Ory token exists and is not expired.
 * Used by the auth provider detection logic.
 */
export function hasValidOryToken(): boolean {
  const token = getOryAccessToken();
  if (!token) return false;

  try {
    // Parse JWT to check expiration
    const parts = token.split(".");
    if (parts.length !== 3) return false;

    const payload = JSON.parse(atob(parts[1]));
    if (!payload.exp) return true; // No expiration claim

    // Add 30 second buffer for clock skew
    return Date.now() < payload.exp * 1000 - 30000;
  } catch {
    return false;
  }
}
