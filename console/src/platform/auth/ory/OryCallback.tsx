/**
 * Ory OAuth2 Callback Handler
 *
 * Handles the OAuth2 Authorization Code callback from Ory.
 * Exchanges the authorization code for tokens and redirects to the original destination.
 */

import { Center, Spinner, Text, VStack } from "@chakra-ui/react";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { setOryAccessToken } from "~/api/oryToken";
import {
  persistLoginEmail,
  persistProviderChoice,
} from "~/auth/detectAuthProvider";
import { getOryUserManager } from "~/auth/oryOidcConfig";

/**
 * OAuth2 callback page that completes the authentication flow.
 *
 * This component:
 * 1. Receives the authorization code from Ory via URL params
 * 2. Exchanges it for tokens via oidc-client-ts
 * 3. Stores the access token (JWT) for API authentication
 * 4. Redirects to the original destination
 */
export const OryCallback = () => {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const userManager = getOryUserManager();

        // Complete the OAuth2 flow - exchanges code for tokens
        const user = await userManager.signinRedirectCallback();

        if (!user) {
          throw new Error("No user returned from authentication");
        }

        // Store the audience-scoped access token for API calls; the
        // backend rejects ID tokens (they are addressed to this client,
        // not the API).
        setOryAccessToken(user.access_token);

        // Persist that user authenticated with Ory for sticky routing,
        // plus their email so future visits can run email-first discovery
        // even after the sticky choice is cleared (e.g. by logout).
        persistProviderChoice("ory");
        if (user.profile.email) {
          persistLoginEmail(user.profile.email);
        }

        // Get the original destination (stored before redirect)
        const returnTo = sessionStorage.getItem("ory_return_to") ?? "/";
        sessionStorage.removeItem("ory_return_to");

        // Navigate to the original destination
        navigate(returnTo, { replace: true });
      } catch (err) {
        console.error("OAuth callback failed:", err);
        setError(
          err instanceof Error ? err.message : "Authentication callback failed",
        );
      }
    };

    handleCallback();
  }, [navigate]);

  if (error) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Text fontSize="xl" fontWeight="bold">
            Authentication Error
          </Text>
          <Text color="red.500">{error}</Text>
          <Text
            as="button"
            color="blue.500"
            textDecoration="underline"
            cursor="pointer"
            onClick={() => navigate("/")}
          >
            Go Home
          </Text>
        </VStack>
      </Center>
    );
  }

  return (
    <Center h="100vh">
      <VStack spacing={4}>
        <Spinner size="xl" />
        <Text>Completing authentication...</Text>
      </VStack>
    </Center>
  );
};

export default OryCallback;
