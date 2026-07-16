/**
 * Hook to access Ory authentication context.
 *
 * This is separated from OryProviderWrapper.tsx to avoid
 * React Fast Refresh warnings about mixed exports.
 */

import { createContext, useContext } from "react";

import type { OrySession } from "~/external-library-wrappers/ory";

// Ory authentication context - matches the shape we'll need for the real implementation
export interface OryAuthContextValue {
  session: OrySession | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => void;
  logout: () => void;
  error: Error | null;
}

export const OryAuthContext = createContext<OryAuthContextValue | null>(null);

/**
 * Hook to access Ory authentication context.
 * Throws if used outside OryProviderWrapper.
 */
export function useOryAuth(): OryAuthContextValue {
  const context = useContext(OryAuthContext);
  if (!context) {
    throw new Error("useOryAuth must be used within OryProviderWrapper");
  }
  return context;
}
