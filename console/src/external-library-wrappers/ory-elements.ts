/* eslint-disable no-restricted-imports */
/**
 * This file is a facade for the @ory/elements-react library.
 * It provides pre-built React components for Ory authentication flows.
 *
 * All Ory Elements imports should go through this file.
 * ESLint enforces this via no-restricted-imports.
 *
 * For Ory API client imports, use ~/external-library-wrappers/ory.ts instead.
 */

// Core provider and configuration
export {
  OryConfigurationProvider,
  OryProvider,
  type OryProviderProps,
  useComponents,
  useOryConfiguration,
  useOryFlow,
} from "@ory/elements-react";

// Card components
export {
  OryCard,
  OryCardContent,
  OryCardFooter,
  OryCardHeader,
  OryCardValidationMessages,
  OryConsentCard,
  OrySelfServiceFlowCard,
} from "@ory/elements-react";

// Form components
export {
  OryForm,
  OryFormGroupDivider,
  OryFormSsoButtons,
  OryFormSsoForm,
  OrySettingsFormSection,
} from "@ory/elements-react";

// Settings components
export { OrySettingsCard } from "@ory/elements-react";

// Generic components
export { OryPageHeader } from "@ory/elements-react";

// Utilities
export {
  messageTestId,
  OryLocales,
  uiTextToFormattedMessage,
  useNodeSorter,
} from "@ory/elements-react";

// Session management (client-side)
export {
  type SessionContextData,
  SessionProvider,
  type SessionProviderProps,
  useSession,
} from "@ory/elements-react/client";

// Default theme flow components
// These are the main pre-built UI components for each auth flow
export {
  Consent,
  Login,
  Error as OryError,
  Recovery,
  Registration,
  Settings,
  Verification,
} from "@ory/elements-react/theme";

// Re-export common types for convenience
export type {
  OryClientConfiguration,
  OryFlowComponentOverrides,
  OryFlowComponents,
} from "@ory/elements-react";

// Note: Import styles in your app entry point:
// import "@ory/elements-react/theme/styles.css"
