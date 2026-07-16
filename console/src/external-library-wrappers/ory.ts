/* eslint-disable no-restricted-imports */
/**
 * This file is a facade for the @ory/client and @ory/client-fetch libraries.
 * It is used primarily to mock the Ory library in tests via `vi.mock` in ~/vitest.setup.ts.
 * Make sure anything you'd like to mock is updated in ./__mocks__/ory.ts
 *
 * All Ory client imports should go through this file. ESLint enforces this via no-restricted-imports.
 * For UI components, use ~/external-library-wrappers/ory-elements.ts instead.
 */

// Legacy @ory/client exports (for existing code)
export {
  Configuration,
  FrontendApi,
  type Identity,
  type IdentityCredentials,
  type LoginFlow,
  type LogoutFlow,
  OAuth2Api,
  type RecoveryFlow,
  type RegistrationFlow,
  type Session,
  type SettingsFlow,
  type VerificationFlow,
} from "@ory/client";

// @ory/client-fetch exports (used by @ory/elements-react)
// These are the newer fetch-based API clients
export {
  Configuration as FetchConfiguration,
  FrontendApi as FetchFrontendApi,
  type Identity as FetchIdentity,
  type LoginFlow as FetchLoginFlow,
  OAuth2Api as FetchOAuth2Api,
  type RecoveryFlow as FetchRecoveryFlow,
  type RegistrationFlow as FetchRegistrationFlow,
  type Session as FetchSession,
  type SettingsFlow as FetchSettingsFlow,
  type VerificationFlow as FetchVerificationFlow,
  FlowType,
} from "@ory/client-fetch";

// Type aliases for clearer usage throughout the codebase
export type {
  Identity as OryIdentity,
  Session as OrySession,
} from "@ory/client";
