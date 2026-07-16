/**
 * Ory Login Page
 *
 * Renders the Ory Elements Login component for user authentication.
 * This page handles the login flow including password, social login, and MFA.
 */

import { Box, Center, Spinner, Text, VStack } from "@chakra-ui/react";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { persistProviderChoice } from "~/auth/detectAuthProvider";
import { FetchLoginFlow } from "~/external-library-wrappers/ory";
import { Login } from "~/external-library-wrappers/ory-elements";

import { buildOryClientConfig, createOryFrontendApi } from "./oryConfig";

export interface OryLoginPageProps {
  /** URL to redirect to after successful login */
  redirectUrl?: string;
}

export const OryLoginPage = ({ redirectUrl }: OryLoginPageProps) => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [flow, setFlow] = useState<FetchLoginFlow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Get redirect URL from props or query params
  const returnTo =
    redirectUrl ||
    searchParams.get("redirectUrl") ||
    searchParams.get("return_to") ||
    "/";

  const oryClient = createOryFrontendApi();
  const config = buildOryClientConfig();

  // Initialize or fetch the login flow
  const initializeFlow = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if there's a flow ID in the URL (returning from Ory)
      const flowId = searchParams.get("flow");

      let loginFlow: FetchLoginFlow;

      if (flowId) {
        // Fetch existing flow
        loginFlow = await oryClient.getLoginFlow({ id: flowId });
      } else {
        // Create new flow
        loginFlow = await oryClient.createBrowserLoginFlow({
          returnTo,
        });
      }

      setFlow(loginFlow);
    } catch (err) {
      console.error("Failed to initialize login flow:", err);
      setError(
        err instanceof Error ? err.message : "Failed to initialize login",
      );
    } finally {
      setIsLoading(false);
    }
  }, [oryClient, searchParams, returnTo]);

  useEffect(() => {
    initializeFlow();
  }, [initializeFlow]);

  // Handle successful login
  const handleLoginSuccess = useCallback(() => {
    // Persist that user authenticated with Ory for sticky routing
    persistProviderChoice("ory");

    // Navigate to the return URL
    navigate(returnTo, { replace: true });
  }, [navigate, returnTo]);

  // Monitor flow state for completion
  useEffect(() => {
    if (flow?.return_to && flow.return_to !== returnTo) {
      // Flow completed, redirect
      handleLoginSuccess();
    }
  }, [flow, returnTo, handleLoginSuccess]);

  if (isLoading) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading login...</Text>
        </VStack>
      </Center>
    );
  }

  if (error) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Text color="red.500">Error: {error}</Text>
          <Text
            as="button"
            color="blue.500"
            textDecoration="underline"
            onClick={() => initializeFlow()}
          >
            Try again
          </Text>
        </VStack>
      </Center>
    );
  }

  if (!flow) {
    return (
      <Center h="100vh">
        <Text>Unable to load login flow</Text>
      </Center>
    );
  }

  return (
    <Box minH="100vh" bg="gray.50" py={12}>
      <Center>
        <Box maxW="md" w="full">
          {/* Cast flow due to @ory/client-fetch version mismatch between deps */}
          <Login flow={flow as never} config={config} />
        </Box>
      </Center>
    </Box>
  );
};

export default OryLoginPage;
