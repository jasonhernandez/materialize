/**
 * Ory Invite Acceptance Page
 *
 * Landing page for organization invite links
 * (`/auth/invite?token=...`, built by ~/api/orgMembers.buildInviteLink).
 *
 * Invited signups must reach the cloud registration prehook with the
 * invite token in the flow's `transient_payload` — that is what routes
 * the new identity into the inviter's organization instead of minting a
 * fresh one. Ory's hosted Account Experience cannot carry a transient
 * payload, so this page drives the Kratos browser registration flow
 * directly: create the flow (which sets the CSRF cookie), render our own
 * form, and submit the password method as JSON with
 * `transient_payload.invite_token`.
 *
 * On success the browser is sent into the normal OAuth2 login
 * (`/?auth_provider=ory`); if the project issued a Kratos session on
 * registration the hosted login step is skipped silently.
 */

import {
  Box,
  Button,
  Center,
  FormControl,
  Heading,
  Input,
  Spinner,
  Text,
  VStack,
} from "@chakra-ui/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { useSearchParams } from "react-router-dom";

import {
  persistLoginEmail,
  persistProviderChoice,
} from "~/auth/detectAuthProvider";
import Alert from "~/components/Alert";
import { LabeledInput } from "~/components/formComponentsV2";
import { FetchRegistrationFlow } from "~/external-library-wrappers/ory";

import { createOryFrontendApi } from "./oryConfig";

interface InviteFormState {
  email: string;
  password: string;
}

/** The CSRF token Kratos embeds in every browser flow's UI nodes. */
function csrfTokenFromFlow(flow: FetchRegistrationFlow): string | undefined {
  for (const node of flow.ui.nodes) {
    const attributes = node.attributes as {
      name?: string;
      value?: string;
    };
    if (attributes.name === "csrf_token") {
      return attributes.value;
    }
  }
  return undefined;
}

/**
 * Human-readable messages from a Kratos flow response: flow-level
 * messages first (this is where the registration prehook's deny text —
 * wrong email, consumed/expired invite — arrives), then field-level ones
 * (e.g. password policy violations).
 */
function messagesFromFlow(flow: {
  ui?: {
    messages?: Array<{ text: string }>;
    nodes?: Array<{ messages?: Array<{ text: string }> }>;
  };
}): string[] {
  const messages: string[] = [];
  for (const message of flow.ui?.messages ?? []) {
    messages.push(message.text);
  }
  for (const node of flow.ui?.nodes ?? []) {
    for (const message of node.messages ?? []) {
      messages.push(message.text);
    }
  }
  return messages;
}

export const OryInvitePage = () => {
  const [searchParams] = useSearchParams();
  const inviteToken = searchParams.get("token");

  const [flow, setFlow] = useState<FetchRegistrationFlow | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const { register, handleSubmit, formState } = useForm<InviteFormState>({
    mode: "onTouched",
  });

  // Memoized: a fresh client every render would change the identity of
  // the flow-fetch callback and re-trigger its effect (the flow-page
  // infinite-loop bug class).
  const oryClient = useMemo(() => createOryFrontendApi(), []);

  const initializeFlow = useCallback(async () => {
    setInitError(null);
    try {
      const registrationFlow = await oryClient.createBrowserRegistrationFlow(
        {},
      );
      setFlow(registrationFlow);
    } catch (err) {
      console.error("Failed to initialize the invite registration flow:", err);
      setInitError(
        "We could not start the signup flow. Please try again in a moment.",
      );
    }
  }, [oryClient]);

  useEffect(() => {
    if (inviteToken) {
      initializeFlow();
    }
  }, [inviteToken, initializeFlow]);

  const onSubmit = useCallback(
    async (values: InviteFormState) => {
      if (!flow || !inviteToken) return;
      setSubmitting(true);
      setFormError(null);
      try {
        await oryClient.updateRegistrationFlow({
          flow: flow.id,
          updateRegistrationFlowBody: {
            method: "password",
            csrf_token: csrfTokenFromFlow(flow),
            password: values.password,
            traits: { email: values.email },
            // What makes this an *invited* signup: the cloud registration
            // prehook validates the token and stamps the inviter's
            // organization onto the new identity.
            transient_payload: { invite_token: inviteToken },
          },
        });
        // Registered. Remember the provider + email and enter the normal
        // OAuth2 login; a registration-issued Kratos session makes the
        // hosted login step a silent redirect.
        persistProviderChoice("ory");
        persistLoginEmail(values.email);
        window.location.assign("/?auth_provider=ory");
      } catch (err: unknown) {
        const response = (err as { response?: Response }).response;
        if (!response) {
          console.error("Invite registration failed:", err);
          setFormError("Signup failed. Please try again.");
          setSubmitting(false);
          return;
        }
        const body = await response.json().catch(() => null);
        // 422: Kratos wants a browser redirect to finish (e.g. into an
        // OAuth2 continuation) — that is success for our purposes.
        if (body?.redirect_browser_to) {
          persistProviderChoice("ory");
          persistLoginEmail(values.email);
          window.location.assign(body.redirect_browser_to);
          return;
        }
        if (body?.error?.id === "session_already_available") {
          setFormError(
            "You are already signed in. Sign out first, then open the invite link again.",
          );
        } else if (body?.error?.id === "self_service_flow_expired") {
          setFormError("The signup flow expired. Please try again.");
          initializeFlow();
        } else if (body?.ui) {
          // A fresh flow with messages: validation errors, or the
          // registration prehook's denial (wrong email for this invite,
          // consumed or expired invite).
          const messages = messagesFromFlow(body);
          setFormError(
            messages.length > 0
              ? messages.join(" ")
              : "Signup failed. Please check your details and try again.",
          );
          // Kratos returned the flow's next state; keep using it so the
          // CSRF token stays valid.
          setFlow(body as FetchRegistrationFlow);
        } else {
          setFormError(
            body?.error?.message ?? "Signup failed. Please try again.",
          );
        }
        setSubmitting(false);
      }
    },
    [flow, inviteToken, oryClient, initializeFlow],
  );

  if (!inviteToken) {
    return (
      <Center h="100vh">
        <VStack spacing={4} maxW="md" textAlign="center">
          <Heading size="md">Invalid invite link</Heading>
          <Text>
            This invite link is missing its token. Ask your administrator to
            send you a new invite link.
          </Text>
        </VStack>
      </Center>
    );
  }

  if (initError) {
    return (
      <Center h="100vh">
        <VStack spacing={4} maxW="md" textAlign="center">
          <Alert variant="error" message={initError} />
          <Button variant="primary" onClick={() => initializeFlow()}>
            Try again
          </Button>
        </VStack>
      </Center>
    );
  }

  if (!flow) {
    return (
      <Center h="100vh">
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading your invitation…</Text>
        </VStack>
      </Center>
    );
  }

  return (
    <Center minH="100vh" py={12}>
      <Box maxW="md" w="full" px={6}>
        <form onSubmit={handleSubmit(onSubmit)}>
          <VStack spacing={6} alignItems="stretch">
            <VStack spacing={2} alignItems="start">
              <Heading size="lg">Join your team on Materialize</Heading>
              <Text textStyle="text-base">
                You have been invited to join an organization. Create your
                account with the email address the invitation was sent to.
              </Text>
            </VStack>
            {formError && (
              <Alert variant="error" minWidth="100%" message={formError} />
            )}
            <FormControl isInvalid={!!formState.errors.email}>
              <LabeledInput
                label="Email"
                error={formState.errors.email?.message}
                variant="stretch"
              >
                <Input
                  {...register("email", {
                    required: "Email is required.",
                    pattern: {
                      value: /@/,
                      message: "Enter a valid email address.",
                    },
                  })}
                  autoCorrect="off"
                  autoComplete="email"
                  placeholder="you@company.com"
                  size="lg"
                  variant={formState.errors.email ? "error" : "default"}
                />
              </LabeledInput>
            </FormControl>
            <FormControl isInvalid={!!formState.errors.password}>
              <LabeledInput
                label="Password"
                error={formState.errors.password?.message}
                variant="stretch"
              >
                <Input
                  {...register("password", {
                    required: "Password is required.",
                    minLength: {
                      value: 8,
                      message: "Password must be at least 8 characters.",
                    },
                  })}
                  autoCorrect="off"
                  autoComplete="new-password"
                  type="password"
                  placeholder="Choose a password"
                  size="lg"
                  variant={formState.errors.password ? "error" : "default"}
                />
              </LabeledInput>
            </FormControl>
            <Button
              variant="primary"
              size="lg"
              type="submit"
              isLoading={submitting}
              spinner={<Spinner />}
              width="100%"
            >
              Accept invitation
            </Button>
          </VStack>
        </form>
      </Box>
    </Center>
  );
};

export default OryInvitePage;
