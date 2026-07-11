// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Spike evaluating Ory Elements (@ory/elements-react) for the Ory login
 * and account UX (cloud/doc/design/20260710_ory_auth_migration.md).
 * Renders the Kratos self-service flows with Elements' default theme,
 * re-branded via CSS variables (orySpikeTheme.css). See
 * doc/ory-elements-spike.md for findings.
 *
 * Only reachable when `mz-ory-spike-sdk-url` is set in localStorage (see
 * orySpikeConfig.ts). Not linked from anywhere in the product UI.
 */

import "@ory/elements-react/theme/styles.css";
import "./orySpikeTheme.css";

import {
  Box,
  Button,
  chakra,
  HStack,
  Spinner,
  Text,
  useTheme,
  VStack,
} from "@chakra-ui/react";
import {
  type FrontendApi,
  type LoginFlow,
  type RecoveryFlow,
  type RegistrationFlow,
  ResponseError,
  type SettingsFlow,
  type VerificationFlow,
} from "@ory/client-fetch";
import {
  Login,
  Recovery,
  Registration,
  Settings,
  Verification,
} from "@ory/elements-react/theme";
import { useMutation, useQuery } from "@tanstack/react-query";
import React from "react";
import { Link, Route, useSearchParams } from "react-router-dom";

import Alert from "~/components/Alert";
import { SentryRoutes } from "~/sentry";
import { MaterializeTheme } from "~/theme";

import {
  buildOryClientConfiguration,
  buildOryFrontendClient,
  getOrySpikeSdkUrl,
  ORY_SPIKE_BASE_PATH,
} from "./orySpikeConfig";

const useOrySpike = () => {
  // The routes only mount when the SDK URL is configured, so this is
  // always defined in practice. The empty-string fallback keeps hook
  // order stable if the localStorage key is removed mid-session.
  const sdkUrl = getOrySpikeSdkUrl() ?? "";
  return React.useMemo(
    () => ({
      client: buildOryFrontendClient(sdkUrl),
      config: buildOryClientConfiguration(sdkUrl),
    }),
    [sdkUrl],
  );
};

/**
 * Fetches the self-service flow named by the `?flow=` search param, or
 * creates a fresh browser flow when the param is absent or the flow is
 * expired. The created flow's id is written back to the URL so refreshes
 * and Kratos error redirects resume it.
 */
const useOryFlowQuery = <F extends { id: string }>(
  flowName: string,
  create: (client: FrontendApi) => Promise<F>,
  get: (client: FrontendApi, id: string) => Promise<F>,
) => {
  const { client, config } = useOrySpike();
  const [searchParams, setSearchParams] = useSearchParams();
  const flowId = searchParams.get("flow") ?? undefined;

  // The client and the create/get callbacks are fully determined by the
  // SDK URL and flowName, so the key below is the complete cache identity.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const query = useQuery({
    queryKey: ["ory-spike", config.sdk?.url, flowName, flowId],
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
    queryFn: async () => {
      if (flowId) {
        try {
          return await get(client, flowId);
        } catch {
          // Expired, consumed, or foreign flow id. Start fresh below.
        }
      }
      return await create(client);
    },
  });

  const createdFlowId = query.data?.id;
  React.useEffect(() => {
    if (!createdFlowId || createdFlowId === flowId) return;
    setSearchParams({ flow: createdFlowId }, { replace: true });
  }, [createdFlowId, flowId, setSearchParams]);

  return query;
};

const SpikeNav = () => {
  const { colors } = useTheme<MaterializeTheme>();
  const links: Array<[string, string]> = [
    ["Session", ORY_SPIKE_BASE_PATH],
    ["Login", `${ORY_SPIKE_BASE_PATH}/login`],
    ["Registration", `${ORY_SPIKE_BASE_PATH}/registration`],
    ["Recovery", `${ORY_SPIKE_BASE_PATH}/recovery`],
    ["Verification", `${ORY_SPIKE_BASE_PATH}/verification`],
    ["Settings", `${ORY_SPIKE_BASE_PATH}/settings`],
  ];
  return (
    <HStack spacing="4">
      {links.map(([label, to]) => (
        <chakra.span key={to}>
          <Link to={to} style={{ color: colors.accent.brightPurple }}>
            {label}
          </Link>
        </chakra.span>
      ))}
    </HStack>
  );
};

const SpikeShell = ({ children }: React.PropsWithChildren) => {
  const { colors } = useTheme<MaterializeTheme>();
  return (
    <Box height="100vh" overflowY="auto" background={colors.background.primary}>
      <VStack spacing="8" py="16" px="4">
        <Text textStyle="text-ui-med" color={colors.foreground.secondary}>
          Ory Elements spike (not a product surface)
        </Text>
        <SpikeNav />
        <Box width="100%" maxWidth="480px">
          {children}
        </Box>
      </VStack>
    </Box>
  );
};

const FlowStateFallback = ({
  isLoading,
  error,
}: {
  isLoading: boolean;
  error: unknown;
}) => {
  if (isLoading) {
    return (
      <HStack justifyContent="center" py="8">
        <Spinner />
      </HStack>
    );
  }
  return (
    <Alert
      variant="error"
      minWidth="100%"
      message={
        error instanceof Error
          ? error.message
          : "Failed to initialize the flow. Is the Ory SDK URL reachable and CORS configured for this origin?"
      }
    />
  );
};

const LoginPage = () => {
  const { config } = useOrySpike();
  const {
    data: flow,
    isLoading,
    error,
  } = useOryFlowQuery<LoginFlow>(
    "login",
    (client) =>
      client.createBrowserLoginFlow({
        returnTo: config.project.default_redirect_url,
      }),
    (client, id) => client.getLoginFlow({ id }),
  );

  return (
    <SpikeShell>
      {flow ? (
        <Login flow={flow} config={config} />
      ) : (
        <FlowStateFallback isLoading={isLoading} error={error} />
      )}
    </SpikeShell>
  );
};

const RegistrationPage = () => {
  const { config } = useOrySpike();
  const {
    data: flow,
    isLoading,
    error,
  } = useOryFlowQuery<RegistrationFlow>(
    "registration",
    (client) =>
      client.createBrowserRegistrationFlow({
        returnTo: config.project.default_redirect_url,
      }),
    (client, id) => client.getRegistrationFlow({ id }),
  );

  return (
    <SpikeShell>
      {flow ? (
        <Registration flow={flow} config={config} />
      ) : (
        <FlowStateFallback isLoading={isLoading} error={error} />
      )}
    </SpikeShell>
  );
};

const RecoveryPage = () => {
  const { config } = useOrySpike();
  const {
    data: flow,
    isLoading,
    error,
  } = useOryFlowQuery<RecoveryFlow>(
    "recovery",
    (client) =>
      client.createBrowserRecoveryFlow({
        returnTo: config.project.default_redirect_url,
      }),
    (client, id) => client.getRecoveryFlow({ id }),
  );

  return (
    <SpikeShell>
      {flow ? (
        <Recovery flow={flow} config={config} />
      ) : (
        <FlowStateFallback isLoading={isLoading} error={error} />
      )}
    </SpikeShell>
  );
};

const VerificationPage = () => {
  const { config } = useOrySpike();
  const {
    data: flow,
    isLoading,
    error,
  } = useOryFlowQuery<VerificationFlow>(
    "verification",
    (client) =>
      client.createBrowserVerificationFlow({
        returnTo: config.project.default_redirect_url,
      }),
    (client, id) => client.getVerificationFlow({ id }),
  );

  return (
    <SpikeShell>
      {flow ? (
        <Verification flow={flow} config={config} />
      ) : (
        <FlowStateFallback isLoading={isLoading} error={error} />
      )}
    </SpikeShell>
  );
};

const isUnauthenticatedError = (error: unknown) =>
  error instanceof ResponseError &&
  (error.response.status === 401 || error.response.status === 403);

const SettingsPage = () => {
  const { config } = useOrySpike();
  const {
    data: flow,
    isLoading,
    error,
  } = useOryFlowQuery<SettingsFlow>(
    "settings",
    (client) =>
      client.createBrowserSettingsFlow({
        returnTo: config.project.default_redirect_url,
      }),
    (client, id) => client.getSettingsFlow({ id }),
  );

  return (
    <SpikeShell>
      {flow ? (
        <Settings flow={flow} config={config} />
      ) : isUnauthenticatedError(error) ? (
        <Alert
          variant="info"
          minWidth="100%"
          message="Settings requires an active Ory session. Sign in via the Login flow first."
        />
      ) : (
        <FlowStateFallback isLoading={isLoading} error={error} />
      )}
    </SpikeShell>
  );
};

const SessionPage = () => {
  const { client, config } = useOrySpike();
  // The client is fully determined by the SDK URL in the key.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const sessionQuery = useQuery({
    queryKey: ["ory-spike", config.sdk?.url, "session"],
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: () => client.toSession(),
  });
  const { data: session, isLoading, error } = sessionQuery;

  const { mutate: logout, isPending: isLoggingOut } = useMutation({
    mutationFn: async () => {
      const { logout_url } = await client.createBrowserLogoutFlow({
        returnTo: config.project.login_ui_url,
      });
      window.location.assign(logout_url);
    },
  });

  const { colors } = useTheme<MaterializeTheme>();

  return (
    <SpikeShell>
      {isLoading ? (
        <FlowStateFallback isLoading error={undefined} />
      ) : session ? (
        <VStack alignItems="stretch" spacing="4">
          <Text textStyle="text-ui-med">Active Ory session</Text>
          <chakra.pre
            fontSize="xs"
            overflowX="auto"
            p="4"
            borderRadius="lg"
            border="1px solid"
            borderColor={colors.border.primary}
          >
            {JSON.stringify(session, undefined, 2)}
          </chakra.pre>
          <Button
            variant="primary"
            size="lg"
            onClick={() => logout()}
            isLoading={isLoggingOut}
          >
            Log out
          </Button>
        </VStack>
      ) : (
        <Alert
          variant="info"
          minWidth="100%"
          message={
            isUnauthenticatedError(error)
              ? "No active Ory session. Sign in via the Login flow."
              : "Could not reach the Ory project. Check the SDK URL and CORS configuration."
          }
        />
      )}
    </SpikeShell>
  );
};

const OrySpikeRoutes = () => (
  <SentryRoutes>
    <Route path="/" element={<SessionPage />} />
    <Route path="login" element={<LoginPage />} />
    <Route path="registration" element={<RegistrationPage />} />
    <Route path="recovery" element={<RecoveryPage />} />
    <Route path="verification" element={<VerificationPage />} />
    <Route path="settings" element={<SettingsPage />} />
  </SentryRoutes>
);

export default OrySpikeRoutes;
