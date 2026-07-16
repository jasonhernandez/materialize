/**
 * Auth Provider Wrapper
 *
 * Detects which authentication provider to use and renders the appropriate
 * provider wrapper (Frontegg or Ory). This component handles the provider
 * detection logic at app startup, before any authentication occurs.
 */

import { useAtom } from "jotai";
import React, { useEffect, useState } from "react";

import LoadingScreen from "~/components/LoadingScreen";
import { appConfig } from "~/config/AppConfig";

import { authProviderAtom, authProviderDetectedAtom } from "./authProviderAtom";
import {
  type AuthProviderType,
  detectAuthProvider,
} from "./detectAuthProvider";

export interface AuthProviderWrapperProps {
  children: React.ReactNode;
  /**
   * Component to render for Frontegg authentication.
   * Should be FronteggProviderWrapper.
   */
  FronteggWrapper: React.ComponentType<{ children: React.ReactNode }>;
  /**
   * Component to render for Ory authentication.
   * Should be OryProviderWrapper.
   */
  OryWrapper: React.ComponentType<{ children: React.ReactNode }>;
}

/**
 * Wrapper component that detects and routes to the appropriate auth provider.
 *
 * This component:
 * 1. Detects which auth provider to use (runs once at startup)
 * 2. Sets the provider in Jotai state
 * 3. Renders the appropriate provider wrapper
 *
 * For non-cloud modes (self-managed, impersonation), it bypasses detection
 * and renders children directly.
 */
export const AuthProviderWrapper = ({
  children,
  FronteggWrapper,
  OryWrapper,
}: AuthProviderWrapperProps) => {
  const [provider, setProvider] = useAtom(authProviderAtom);
  const [detected, setDetected] = useAtom(authProviderDetectedAtom);
  const [isDetecting, setIsDetecting] = useState(!detected);

  useEffect(() => {
    // Skip detection if not in cloud mode or already detected
    if (appConfig.mode !== "cloud" || detected) {
      setIsDetecting(false);
      return;
    }

    // Skip detection if impersonating (uses Frontegg tokens directly)
    if (appConfig.isImpersonating) {
      setProvider("frontegg");
      setDetected(true);
      setIsDetecting(false);
      return;
    }

    let cancelled = false;

    const detect = async () => {
      try {
        const detectedProvider = await detectAuthProvider();
        if (!cancelled) {
          setProvider(detectedProvider);
          setDetected(true);
          setIsDetecting(false);
        }
      } catch (error) {
        // Detection failed, default to Frontegg
        console.error(
          "Auth provider detection failed, defaulting to Frontegg:",
          error,
        );
        if (!cancelled) {
          setProvider("frontegg");
          setDetected(true);
          setIsDetecting(false);
        }
      }
    };

    detect();

    return () => {
      cancelled = true;
    };
  }, [detected, setProvider, setDetected]);

  // Show loading screen while detecting provider
  if (isDetecting) {
    return <LoadingScreen />;
  }

  // For non-cloud modes, render children directly
  if (appConfig.mode !== "cloud") {
    return <>{children}</>;
  }

  // Render the appropriate provider wrapper
  return (
    <AuthProviderRenderer
      provider={provider}
      FronteggWrapper={FronteggWrapper}
      OryWrapper={OryWrapper}
    >
      {children}
    </AuthProviderRenderer>
  );
};

/**
 * Internal component that renders the appropriate provider based on detected type.
 * Separated to avoid re-running detection when children change.
 */
interface AuthProviderRendererProps {
  children: React.ReactNode;
  provider: AuthProviderType;
  FronteggWrapper: React.ComponentType<{ children: React.ReactNode }>;
  OryWrapper: React.ComponentType<{ children: React.ReactNode }>;
}

const AuthProviderRenderer = ({
  children,
  provider,
  FronteggWrapper,
  OryWrapper,
}: AuthProviderRendererProps) => {
  if (provider === "ory") {
    return <OryWrapper>{children}</OryWrapper>;
  }

  return <FronteggWrapper>{children}</FronteggWrapper>;
};
