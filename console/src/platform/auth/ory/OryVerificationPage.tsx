/**
 * Ory Verification Page
 *
 * Renders the Ory Elements Verification component for email verification.
 * This page handles the email verification flow after registration.
 */

import { Box, Center, Spinner, Text, VStack } from "@chakra-ui/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { FetchVerificationFlow } from "~/external-library-wrappers/ory";
import { Verification } from "~/external-library-wrappers/ory-elements";

import { buildOryClientConfig, createOryFrontendApi } from "./oryConfig";

export const OryVerificationPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [flow, setFlow] = useState<FetchVerificationFlow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Memoized: fresh instances every render would change the identity of
  // any callback that closes over them and re-trigger flow-fetch effects.
  const oryClient = useMemo(() => createOryFrontendApi(), []);
  const config = useMemo(() => buildOryClientConfig(), []);

  // Initialize or fetch the verification flow
  const initializeFlow = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if there's a flow ID in the URL (returning from Ory)
      const flowId = searchParams.get("flow");

      let verificationFlow: FetchVerificationFlow;

      if (flowId) {
        // Fetch existing flow
        verificationFlow = await oryClient.getVerificationFlow({ id: flowId });

        // Check if verification is complete
        if (verificationFlow.state === "passed_challenge") {
          // Verification successful, redirect to home
          navigate("/", { replace: true });
          return;
        }
      } else {
        // Create new flow
        verificationFlow = await oryClient.createBrowserVerificationFlow();
      }

      setFlow(verificationFlow);
    } catch (err) {
      console.error("Failed to initialize verification flow:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to initialize verification",
      );
    } finally {
      setIsLoading(false);
    }
  }, [oryClient, searchParams, navigate]);

  useEffect(() => {
    initializeFlow();
  }, [initializeFlow]);

  if (isLoading) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading verification...</Text>
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
        <Text>Unable to load verification</Text>
      </Center>
    );
  }

  return (
    <Box minH="100vh" bg="gray.50" py={12}>
      <Center>
        <Box maxW="md" w="full">
          {/* Cast flow due to @ory/client-fetch version mismatch between deps */}
          <Verification flow={flow as never} config={config} />
        </Box>
      </Center>
    </Box>
  );
};

export default OryVerificationPage;
