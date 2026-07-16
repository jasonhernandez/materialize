/**
 * Ory Registration Page
 *
 * Renders the Ory Elements Registration component for new user signup.
 * This page handles the registration flow including form validation and social signup.
 */

import { Box, Center, Spinner, Text, VStack } from "@chakra-ui/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { persistProviderChoice } from "~/auth/detectAuthProvider";
import { FetchRegistrationFlow } from "~/external-library-wrappers/ory";
import { Registration } from "~/external-library-wrappers/ory-elements";

import { buildOryClientConfig, createOryFrontendApi } from "./oryConfig";

export interface OryRegistrationPageProps {
  /** URL to redirect to after successful registration */
  redirectUrl?: string;
}

export const OryRegistrationPage = ({
  redirectUrl,
}: OryRegistrationPageProps) => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [flow, setFlow] = useState<FetchRegistrationFlow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Get redirect URL from props or query params
  const returnTo =
    redirectUrl ||
    searchParams.get("redirectUrl") ||
    searchParams.get("return_to") ||
    "/";

  // Memoized: fresh instances every render would change the identity of
  // any callback that closes over them and re-trigger flow-fetch effects.
  const oryClient = useMemo(() => createOryFrontendApi(), []);
  const config = useMemo(() => buildOryClientConfig(), []);

  // Initialize or fetch the registration flow
  const initializeFlow = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if there's a flow ID in the URL (returning from Ory)
      const flowId = searchParams.get("flow");

      let registrationFlow: FetchRegistrationFlow;

      if (flowId) {
        // Fetch existing flow
        registrationFlow = await oryClient.getRegistrationFlow({ id: flowId });
      } else {
        // Create new flow
        registrationFlow = await oryClient.createBrowserRegistrationFlow({
          returnTo,
        });
      }

      setFlow(registrationFlow);
    } catch (err) {
      console.error("Failed to initialize registration flow:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to initialize registration",
      );
    } finally {
      setIsLoading(false);
    }
  }, [oryClient, searchParams, returnTo]);

  useEffect(() => {
    initializeFlow();
  }, [initializeFlow]);

  // Handle successful registration
  const handleRegistrationSuccess = useCallback(() => {
    // Persist that user authenticated with Ory for sticky routing
    persistProviderChoice("ory");

    // Navigate to the return URL
    navigate(returnTo, { replace: true });
  }, [navigate, returnTo]);

  // Monitor flow state for completion
  useEffect(() => {
    if (flow?.return_to && flow.return_to !== returnTo) {
      // Flow completed, redirect
      handleRegistrationSuccess();
    }
  }, [flow, returnTo, handleRegistrationSuccess]);

  if (isLoading) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading registration...</Text>
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
        <Text>Unable to load registration flow</Text>
      </Center>
    );
  }

  return (
    <Box minH="100vh" bg="gray.50" py={12}>
      <Center>
        <Box maxW="md" w="full">
          {/* Cast flow due to @ory/client-fetch version mismatch between deps */}
          <Registration flow={flow as never} config={config} />
        </Box>
      </Center>
    </Box>
  );
};

export default OryRegistrationPage;
