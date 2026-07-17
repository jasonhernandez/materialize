/**
 * Ory Provider Wrapper
 *
 * Provides OAuth2 authentication via oidc-client-ts against Ory Network.
 * Uses OAuth2 Authorization Code flow with PKCE to obtain JWTs for API authentication.
 */

import type { User } from "oidc-client-ts";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { clearOryAccessToken, setOryAccessToken } from "~/api/oryToken";
import { getOryUserManager } from "~/auth/oryOidcConfig";
import LoadingScreen from "~/components/LoadingScreen";
import type { OrySession } from "~/external-library-wrappers/ory";
import { OryAuthContext, type OryAuthContextValue } from "~/hooks/useOryAuth";
import { createOryFrontendApi } from "~/platform/auth/ory/oryConfig";

/**
 * Ory Provider Wrapper using oidc-client-ts for OAuth2 authentication.
 *
 * This component:
 * 1. Checks for existing authenticated user on mount
 * 2. If not authenticated, initiates OAuth2 Authorization Code flow
 * 3. Syncs the id_token (JWT) to sessionStorage for API client usage
 * 4. Provides authentication context to children
 */
export const OryProviderWrapper = ({ children }: React.PropsWithChildren) => {
  const [user, setUser] = useState<User | null>(null);
  const [kratosSession, setKratosSession] = useState<OrySession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Check if we're on the callback route (don't redirect during callback)
  // Also check for /callback in case Ory client is configured with that URI
  const isCallbackRoute =
    window.location.pathname.startsWith("/auth/ory/callback") ||
    window.location.pathname === "/callback";

  // Initialize and check for existing user
  useEffect(() => {
    const checkUser = async () => {
      try {
        const userManager = getOryUserManager();
        const existingUser = await userManager.getUser();

        if (existingUser && !existingUser.expired) {
          setUser(existingUser);

          // Sync the access token to token storage for the API client
          setOryAccessToken(existingUser.access_token);

          // Enrich with the real Kratos session: the id_token carries only
          // standard OIDC claims, while whoami exposes the identity's
          // metadata_public (organization_id, stamped at registration by
          // the sync-server prehook). Uses the Kratos session cookie set
          // during the hosted login flow; falls back to id_token claims
          // when unavailable.
          try {
            const session = await createOryFrontendApi().toSession();
            // @ory/client-fetch's Session is structurally compatible with
            // @ory/client's Session (our OrySession alias).
            setKratosSession(session as unknown as OrySession);
          } catch (whoamiErr) {
            console.warn(
              "Ory whoami failed; falling back to id_token claims",
              whoamiErr,
            );
          }
        } else if (!isCallbackRoute) {
          // No valid user and not on callback - initiate login
          // Store current location for redirect after login
          const returnTo = window.location.pathname + window.location.search;
          sessionStorage.setItem("ory_return_to", returnTo);

          await userManager.signinRedirect();
          return; // Don't set loading to false, we're redirecting
        }
      } catch (err) {
        console.error("Ory auth check failed:", err);
        setError(
          err instanceof Error ? err : new Error("Authentication failed"),
        );
      }

      setIsLoading(false);
    };

    checkUser();
  }, [isCallbackRoute]);

  // Track sign-ins completed after mount: OryCallback finishes the code
  // exchange via the same UserManager, which fires userLoaded — without
  // this subscription the provider's state (and isAuthenticated) would
  // stay stale until a full page reload.
  useEffect(() => {
    const userManager = getOryUserManager();
    const onUserLoaded = (loadedUser: User) => {
      setUser(loadedUser);
      setOryAccessToken(loadedUser.access_token);
      createOryFrontendApi()
        .toSession()
        .then((session) => setKratosSession(session as unknown as OrySession))
        .catch((whoamiErr) => {
          console.warn(
            "Ory whoami failed; falling back to id_token claims",
            whoamiErr,
          );
        });
    };
    const onUserUnloaded = () => {
      setUser(null);
      setKratosSession(null);
    };
    userManager.events.addUserLoaded(onUserLoaded);
    userManager.events.addUserUnloaded(onUserUnloaded);
    return () => {
      userManager.events.removeUserLoaded(onUserLoaded);
      userManager.events.removeUserUnloaded(onUserUnloaded);
    };
  }, []);

  // Login handler - initiate OAuth2 flow
  const login = useCallback(() => {
    const userManager = getOryUserManager();
    const returnTo = window.location.pathname + window.location.search;
    sessionStorage.setItem("ory_return_to", returnTo);

    userManager.signinRedirect().catch((err) => {
      console.error("Login redirect failed:", err);
      setError(err instanceof Error ? err : new Error("Login failed"));
    });
  }, []);

  // Logout handler - clear tokens and redirect
  const logout = useCallback(async () => {
    try {
      const userManager = getOryUserManager();

      // Clear our token storage
      clearOryAccessToken();

      // Sign out via OIDC (clears oidc-client-ts state and redirects to Ory logout)
      await userManager.signoutRedirect();
    } catch (err) {
      console.error("Logout failed:", err);
      // Even if OIDC logout fails, clear local state
      clearOryAccessToken();
      setUser(null);
    }
  }, []);

  // Build context value. Prefer the real Kratos session (whoami) — it
  // carries the identity's metadata_public — and fall back to a session
  // synthesized from the OIDC id_token claims.
  const contextValue: OryAuthContextValue = useMemo(
    () => ({
      session:
        kratosSession ??
        (user
          ? ({
              id: user.profile.sub,
              active: !user.expired,
              identity: {
                id: user.profile.sub,
                traits: {
                  email: user.profile.email ?? "",
                  name: {
                    first: user.profile.given_name,
                    last: user.profile.family_name,
                  },
                },
                // The sync-server's OAuth2 token hook stamps the
                // identity's metadata into ID-token claims, so the
                // fallback session carries the real organization id and
                // roles/permissions. The whole profile remains the
                // metadata_public fallback for tokens issued before the
                // hook existed.
                metadata_public:
                  user.profile.metadata_public ?? user.profile,
                metadata_admin: user.profile.metadata_admin,
              },
            } as never) // Cast to satisfy OrySession type
          : null),
      isAuthenticated: !!user && !user.expired,
      isLoading,
      login,
      logout,
      error,
    }),
    [user, kratosSession, isLoading, login, logout, error],
  );

  // Show loading while checking auth state
  if (isLoading) {
    return (
      <OryAuthContext.Provider value={contextValue}>
        <LoadingScreen />
      </OryAuthContext.Provider>
    );
  }

  // Show error state
  if (error && !user) {
    return (
      <OryAuthContext.Provider value={contextValue}>
        <div style={{ padding: "2rem", textAlign: "center" }}>
          <h1>Authentication Error</h1>
          <p style={{ color: "red" }}>{error.message}</p>
          <button
            onClick={login}
            style={{
              marginTop: "1rem",
              padding: "0.5rem 1rem",
              cursor: "pointer",
            }}
          >
            Try Again
          </button>
        </div>
      </OryAuthContext.Provider>
    );
  }

  // Render children when authenticated or on callback route
  return (
    <OryAuthContext.Provider value={contextValue}>
      {children}
    </OryAuthContext.Provider>
  );
};
