/**
 * Mock implementation of the @ory/elements-react library for testing.
 * This mock is registered in vitest.setup.ts via vi.mock("~/external-library-wrappers/ory-elements")
 */

import React from "react";

// Mock session context
export const mockSession = {
  id: "mock-session-id",
  active: true,
  identity: {
    id: "mock-identity-id",
    traits: {
      email: "test@example.com",
      name: { first: "Test", last: "User" },
    },
  },
};

// Mock providers - render children directly
export const OryProvider = ({ children }: { children: React.ReactNode }) =>
  React.createElement(React.Fragment, null, children);

export const OryConfigurationProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => React.createElement(React.Fragment, null, children);

export const SessionProvider = ({ children }: { children: React.ReactNode }) =>
  React.createElement(React.Fragment, null, children);

// Mock hooks
export const useOryConfiguration = () => ({
  sdk: {},
  project: { name: "test-project" },
});

export const useOryFlow = () => ({
  flow: null,
  flowType: null,
});

export const useComponents = () => ({});

export const useSession = () => ({
  session: mockSession,
  isLoading: false,
  initialized: true,
  error: undefined,
  refetch: async () => {},
});

export const useNodeSorter = () => (nodes: unknown[]) => nodes;

// Mock card components
export const OryCard = ({ children }: { children: React.ReactNode }) =>
  React.createElement("div", { "data-testid": "ory-card" }, children);

export const OryCardContent = ({ children }: { children: React.ReactNode }) =>
  React.createElement("div", { "data-testid": "ory-card-content" }, children);

export const OryCardFooter = ({ children }: { children: React.ReactNode }) =>
  React.createElement("div", { "data-testid": "ory-card-footer" }, children);

export const OryCardHeader = ({ children }: { children: React.ReactNode }) =>
  React.createElement("div", { "data-testid": "ory-card-header" }, children);

export const OrySelfServiceFlowCard = () =>
  React.createElement("div", { "data-testid": "ory-self-service-flow-card" });

export const OryConsentCard = () =>
  React.createElement("div", { "data-testid": "ory-consent-card" });

export const OryCardValidationMessages = () =>
  React.createElement("div", { "data-testid": "ory-card-validation-messages" });

// Mock form components
export const OryForm = ({ children }: { children: React.ReactNode }) =>
  React.createElement("form", { "data-testid": "ory-form" }, children);

export const OryFormGroupDivider = () =>
  React.createElement("hr", { "data-testid": "ory-form-divider" });

export const OryFormSsoButtons = () =>
  React.createElement("div", { "data-testid": "ory-form-sso-buttons" });

export const OryFormSsoForm = () =>
  React.createElement("div", { "data-testid": "ory-form-sso-form" });

export const OrySettingsFormSection = ({
  children,
}: {
  children: React.ReactNode;
}) =>
  React.createElement(
    "div",
    { "data-testid": "ory-settings-form-section" },
    children,
  );

// Mock settings components
export const OrySettingsCard = () =>
  React.createElement("div", { "data-testid": "ory-settings-card" });

// Mock generic components
export const OryPageHeader = () =>
  React.createElement("div", { "data-testid": "ory-page-header" });

// Mock utilities
export const OryLocales = {};
export const messageTestId = (id: string) => `ory-message-${id}`;
export const uiTextToFormattedMessage = (text: unknown) => String(text);

// Mock theme flow components
export const Login = () =>
  React.createElement("div", { "data-testid": "ory-login" });

export const Registration = () =>
  React.createElement("div", { "data-testid": "ory-registration" });

export const Recovery = () =>
  React.createElement("div", { "data-testid": "ory-recovery" });

export const Verification = () =>
  React.createElement("div", { "data-testid": "ory-verification" });

export const Settings = () =>
  React.createElement("div", { "data-testid": "ory-settings" });

export const OryError = () =>
  React.createElement("div", { "data-testid": "ory-error" });

export const Consent = () =>
  React.createElement("div", { "data-testid": "ory-consent" });

// Mock types (re-exported as empty objects for type compatibility)
export type OryProviderProps = Record<string, unknown>;
export type SessionContextData = Record<string, unknown>;
export type SessionProviderProps = Record<string, unknown>;
export type OryClientConfiguration = Record<string, unknown>;
export type OryFlowComponentOverrides = Record<string, unknown>;
export type OryFlowComponents = Record<string, unknown>;
