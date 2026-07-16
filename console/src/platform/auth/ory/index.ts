/**
 * Ory Authentication Pages
 *
 * This module exports all Ory-related authentication page components
 * and configuration utilities.
 */

// Page components
export { OryLoginPage, type OryLoginPageProps } from "./OryLoginPage";
export { OryRecoveryPage } from "./OryRecoveryPage";
export {
  OryRegistrationPage,
  type OryRegistrationPageProps,
} from "./OryRegistrationPage";
export { OrySettingsPage, type OrySettingsPageProps } from "./OrySettingsPage";
export { OryVerificationPage } from "./OryVerificationPage";

// Routes
export { OryAuthRoutes } from "./OryAuthRoutes";

// Configuration utilities
export {
  buildOryClientConfig,
  createOryFrontendApi,
  oryFlowUrls,
} from "./oryConfig";
