/**
 * Ory Recovery Page
 *
 * Renders the Ory Elements Recovery component for password recovery.
 * This page handles the "forgot password" flow.
 */

import { Box, Center, Spinner, Text, VStack } from "@chakra-ui/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { FetchRecoveryFlow } from "~/external-library-wrappers/ory";
import { Recovery } from "~/external-library-wrappers/ory-elements";

import { buildOryClientConfig, createOryFrontendApi } from "./oryConfig";

export const OryRecoveryPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [flow, setFlow] = useState<FetchRecoveryFlow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Memoized: fresh instances every render would change the identity of
  // any callback that closes over them and re-trigger flow-fetch effects.
  const oryClient = useMemo(() => createOryFrontendApi(), []);
  const config = useMemo(() => buildOryClientConfig(), []);

  // Initialize or fetch the recovery flow
  const initializeFlow = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if there's a flow ID in the URL (returning from Ory)
      const flowId = searchParams.get("flow");

      let recoveryFlow: FetchRecoveryFlow;

      if (flowId) {
        // Fetch existing flow
        recoveryFlow = await oryClient.getRecoveryFlow({ id: flowId });

        // Check if recovery is complete
        if (recoveryFlow.state === "passed_challenge") {
          // Recovery successful, redirect to settings to set new password
          navigate("/auth/ory/settings?section=password", { replace: true });
          return;
        }
      } else {
        // Create new flow
        recoveryFlow = await oryClient.createBrowserRecoveryFlow();
      }

      setFlow(recoveryFlow);
    } catch (err) {
      console.error("Failed to initialize recovery flow:", err);
      setError(
        err instanceof Error ? err.message : "Failed to initialize recovery",
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
          <Text>Loading password recovery...</Text>
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
        <Text>Unable to load password recovery</Text>
      </Center>
    );
  }

  return (
    <Box minH="100vh" bg="gray.50" py={12}>
      <Center>
        <Box maxW="md" w="full">
          {/* Cast flow due to @ory/client-fetch version mismatch between deps */}
          <Recovery flow={flow as never} config={config} />
        </Box>
      </Center>
    </Box>
  );
};

export default OryRecoveryPage;
