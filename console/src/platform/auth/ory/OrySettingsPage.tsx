/**
 * Ory Settings Page
 *
 * Renders the Ory Elements Settings component for account management.
 * This page handles profile updates, password changes, MFA configuration,
 * and other account settings.
 *
 * This is the Ory equivalent of Frontegg's AdminPortal for user settings.
 */

import {
  Box,
  Center,
  Container,
  Heading,
  Spinner,
  Text,
  VStack,
} from "@chakra-ui/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { FetchSettingsFlow } from "~/external-library-wrappers/ory";
import { Settings } from "~/external-library-wrappers/ory-elements";

import {
  buildOryClientConfig,
  createOryFrontendApi,
  oryFlowUrls,
} from "./oryConfig";

export interface OrySettingsPageProps {
  /** Optional section to scroll to (e.g., "password", "totp", "webauthn") */
  section?: string;
}

export const OrySettingsPage = ({ section }: OrySettingsPageProps) => {
  const [searchParams] = useSearchParams();
  const [flow, setFlow] = useState<FetchSettingsFlow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Memoized: fresh instances every render would change initializeFlow's
  // identity and re-trigger the effect, looping the flow fetch forever.
  const oryClient = useMemo(() => createOryFrontendApi(), []);
  const config = useMemo(() => buildOryClientConfig(), []);

  // Get section from props or query params
  const targetSection = section || searchParams.get("section");

  // Initialize or fetch the settings flow
  const initializeFlow = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if there's a flow ID in the URL (returning from Ory)
      const flowId = searchParams.get("flow");

      let settingsFlow: FetchSettingsFlow;

      if (flowId) {
        // Fetch existing flow
        settingsFlow = await oryClient.getSettingsFlow({ id: flowId });

        // Check if flow was successful (has state = success)
        if (settingsFlow.state === "success") {
          setSuccessMessage("Your settings have been updated successfully.");
          // Create a new flow for further changes
          settingsFlow = await oryClient.createBrowserSettingsFlow();
        }
      } else {
        // Create new flow
        settingsFlow = await oryClient.createBrowserSettingsFlow();
      }

      setFlow(settingsFlow);
    } catch (err) {
      console.error("Failed to initialize settings flow:", err);

      // A 401/403 (or a CORS-level fetch failure, which surfaces as an
      // error with no `response`) means the Kratos session cookie was not
      // sent — the console is on a different site than the Ory project
      // (always the case on localhost; fixed in production by the custom
      // auth domain). The hosted settings flow sees the cookie
      // first-party, so hand off to it rather than dead-ending. The fetch
      // client's ResponseError carries the status on err.response, not
      // the message.
      const response =
        err && typeof err === "object" && "response" in err
          ? (err.response as Response | undefined)
          : undefined;
      const status = response?.status;
      if (status === 401 || status === 403 || response === undefined) {
        window.location.assign(oryFlowUrls.settings());
        return;
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load account settings",
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, [oryClient, searchParams]);

  useEffect(() => {
    initializeFlow();
  }, [initializeFlow]);

  // Scroll to section if specified
  useEffect(() => {
    if (targetSection && !isLoading) {
      const element = document.getElementById(`settings-${targetSection}`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [targetSection, isLoading]);

  if (isLoading) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading account settings...</Text>
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
        <Text>Unable to load account settings</Text>
      </Center>
    );
  }

  return (
    <Box minH="100vh" bg="gray.50" py={8}>
      <Container maxW="container.md">
        <VStack spacing={6} align="stretch">
          <Heading size="lg">Account Settings</Heading>

          {successMessage && (
            <Box
              p={4}
              bg="green.50"
              borderRadius="md"
              borderWidth="1px"
              borderColor="green.200"
            >
              <Text color="green.700">{successMessage}</Text>
            </Box>
          )}

          <Box bg="white" borderRadius="lg" shadow="sm" overflow="hidden">
            {/* Cast flow to any due to @ory/client-fetch version mismatch between our deps and elements-react */}
            <Settings flow={flow as never} config={config} />
          </Box>
        </VStack>
      </Container>
    </Box>
  );
};

export default OrySettingsPage;
