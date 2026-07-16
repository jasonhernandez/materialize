// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import React from "react";
import { Navigate, Route, useLocation } from "react-router-dom";

import { hasActiveSession, LOGIN_PATH } from "~/api/materialize/auth";
import { authProviderAtom } from "~/auth/authProviderAtom";
import { LaunchDarklyProvider } from "~/components/LaunchDarkly";
import LoadingScreen from "~/components/LoadingScreen";
import { type SelfManagedAppConfig } from "~/config/AppConfig";
import { useAppConfig } from "~/config/useAppConfig";
import { useIsAuthenticated } from "~/external-library-wrappers/frontegg";
import {
  hasAuthParams,
  useOidcManagerQuery,
} from "~/external-library-wrappers/oidc";
import { AUTH_ROUTES } from "~/fronteggRoutes";
import { useOryAuth } from "~/hooks/useOryAuth";
import { AuthenticatedRoutes } from "~/platform/AuthenticatedRoutes";
import { SentryRoutes } from "~/sentry";

import { Login } from "./auth/Login";
import { OidcCallback } from "./auth/OidcCallback";
import { OryAuthRoutes } from "./auth/ory/OryAuthRoutes";
import { OryCallback } from "./auth/ory/OryCallback";

// Redirect already-signed-in users off the login page. The password session
// cookie is httpOnly, so probe the server; a live OIDC token skips the probe.
const LoginRoute = () => {
  const { data: oidcManager } = useOidcManagerQuery();
  const hasOidcToken = Boolean(oidcManager?.getIdToken());

  const { data: hasCookieSession } = useQuery({
    queryKey: ["hasActiveSession"],
    queryFn: hasActiveSession,
    enabled: !hasOidcToken,
    staleTime: Infinity,
    retry: false,
  });

  if (hasOidcToken || hasCookieSession) {
    return <Navigate to="/" replace />;
  }
  return <Login />;
};

const OidcAuthGuard = ({ children }: React.PropsWithChildren) => {
  const { isLoading, data: auth } = useOidcManagerQuery();

  // OIDC initialization failed — `OidcProviderWrapper` rendered us without
  // an `AuthProvider` so password sign-in still works. Skip the OIDC checks
  // and let the user reach the app via their password session cookie.
  if (!auth) return <>{children}</>;

  if (isLoading || hasAuthParams()) {
    return <LoadingScreen />;
  }

  // Don't redirect — the user may have a valid password session cookie.
  // The 401 redirect middleware handles expired sessions.
  return children;
};

const SelfManagedRoutes = ({
  appConfig,
}: {
  appConfig: Readonly<SelfManagedAppConfig>;
}) => {
  const isOidc = appConfig.authMode === "Oidc";

  return (
    <SentryRoutes>
      {(appConfig.authMode === "Password" ||
        appConfig.authMode === "Sasl" ||
        isOidc) && <Route path={LOGIN_PATH} element={<LoginRoute />} />}
      {isOidc && <Route path="/auth/callback" element={<OidcCallback />} />}
      <Route
        path="*"
        element={
          isOidc ? (
            <OidcAuthGuard>
              <AuthenticatedRoutes />
            </OidcAuthGuard>
          ) : (
            <AuthenticatedRoutes />
          )
        }
      />
    </SentryRoutes>
  );
};

const CloudAuthenticatedRoutes = () => {
  return (
    <LaunchDarklyProvider>
      <AuthenticatedRoutes />
    </LaunchDarklyProvider>
  );
};

const CloudFronteggAuthenticatedRoutes = () => {
  const isAuthenticated = useIsAuthenticated();

  if (!isAuthenticated) {
    const fullPath = location.pathname + location.search + location.hash;
    const redirectUrl = encodeURIComponent(fullPath);
    return (
      <Navigate to={`${AUTH_ROUTES.loginPath}?redirectUrl=${redirectUrl}`} />
    );
  }

  return <CloudAuthenticatedRoutes />;
};

/**
 * Ory-specific routes container.
 * Handles both auth flows (login, registration, etc.) and authenticated routes.
 *
 * Auth flow routes (/auth/ory/* and /callback) are accessible without authentication.
 * Other routes require authentication and redirect to login if not authenticated.
 */
const CloudOryRoutes = () => {
  const { isAuthenticated, isLoading } = useOryAuth();
  // Must come from the router (not window.location) so this component
  // re-renders on SPA navigation: after OryCallback navigates away from
  // /callback, isAuthRoute has to be re-evaluated or the stale
  // auth-routes-only table matches nothing and renders a blank page.
  const routerLocation = useLocation();

  // Check if we're on an Ory auth route (these don't require authentication)
  // Also check /callback for OAuth2 redirect compatibility
  const isAuthRoute =
    routerLocation.pathname.startsWith("/auth/ory/") ||
    routerLocation.pathname === "/callback";

  // Auth routes are always accessible
  if (isAuthRoute) {
    return (
      <SentryRoutes>
        <Route path="/auth/ory/*" element={<OryAuthRoutes />} />
        <Route path="/callback" element={<OryCallback />} />
      </SentryRoutes>
    );
  }

  // While Ory is loading for non-auth routes, show loading state
  if (isLoading) {
    return null;
  }

  // For non-auth routes, check authentication
  if (!isAuthenticated) {
    const fullPath =
      routerLocation.pathname + routerLocation.search + routerLocation.hash;
    const redirectUrl = encodeURIComponent(fullPath);
    return <Navigate to={`/auth/ory/login?redirectUrl=${redirectUrl}`} />;
  }

  return <CloudAuthenticatedRoutes />;
};

export const UnauthenticatedRoutes = () => {
  const appConfig = useAppConfig();
  const authProvider = useAtomValue(authProviderAtom);

  if (appConfig.mode === "self-managed") {
    return <SelfManagedRoutes appConfig={appConfig} />;
  }
  // We assume impersonation users are already authenticated before they load the Console.

  if (appConfig.mode === "cloud" && appConfig.isImpersonating) {
    return <CloudAuthenticatedRoutes />;
  }

  // Route to appropriate auth provider based on detection
  if (authProvider === "ory") {
    return <CloudOryRoutes />;
  }

  return <CloudFronteggAuthenticatedRoutes />;
};
