/**
 * Mock implementation of the Ory library for testing.
 * This mock is registered in vitest.setup.ts via vi.mock("~/external-library-wrappers/ory")
 */

/* eslint-disable no-restricted-imports */
import type { Identity, Session } from "@ory/client";

// Re-export types (these don't need mocking)
export type { Identity, Session } from "@ory/client";
/* eslint-enable no-restricted-imports */
export type OryIdentity = Identity;
export type OrySession = Session;

// Mock identity with typical Materialize user structure
export const mockOryIdentity: Identity = {
  id: "ory-mock-user-id",
  schema_id: "preset://email",
  schema_url: "https://example.com/schemas/email.json",
  state: "active",
  traits: {
    email: "user@example.com",
    name: {
      first: "Test",
      last: "User",
    },
  },
  metadata_admin: {
    roles: ["MaterializePlatformAdmin"],
    permissions: [
      "materialize.environment.read",
      "materialize.environment.write",
      "fe.secure.read.tenantApiTokens",
      "fe.secure.write.tenantApiTokens",
      "fe.secure.delete.tenantApiTokens",
      "materialize.invoice.read",
    ],
    frontegg_user_id: "original-frontegg-uuid",
    migrated_at: "2025-01-01T00:00:00Z",
  },
  metadata_public: {
    tenant_id: "tenant-id",
    onboarded: true,
  },
  created_at: "2025-01-01T00:00:00Z",
  updated_at: "2025-01-01T00:00:00Z",
};

// Mock session
export const mockOrySession: Session = {
  id: "ory-mock-session-id",
  active: true,
  identity: mockOryIdentity,
  authenticated_at: new Date().toISOString(),
  expires_at: new Date(Date.now() + 3600000).toISOString(),
  issued_at: new Date().toISOString(),
};

// Mock access token for API calls
export const MOCK_ORY_ACCESS_TOKEN = "mock-ory-access-token";

// Mock Configuration class
export const Configuration = vi.fn().mockImplementation(() => ({
  basePath: "https://mock-ory-project.projects.oryapis.com",
}));

// Mock FrontendApi class
export const FrontendApi = vi.fn().mockImplementation(() => ({
  toSession: vi.fn().mockResolvedValue({ data: mockOrySession }),
  createBrowserLoginFlow: vi.fn().mockResolvedValue({
    data: {
      id: "login-flow-id",
      request_url:
        "https://mock-ory-project.projects.oryapis.com/self-service/login/browser",
    },
  }),
  createBrowserLogoutFlow: vi.fn().mockResolvedValue({
    data: {
      logout_url:
        "https://mock-ory-project.projects.oryapis.com/self-service/logout",
      logout_token: "logout-token",
    },
  }),
  createBrowserRegistrationFlow: vi.fn().mockResolvedValue({
    data: {
      id: "registration-flow-id",
      request_url:
        "https://mock-ory-project.projects.oryapis.com/self-service/registration/browser",
    },
  }),
  createBrowserRecoveryFlow: vi.fn().mockResolvedValue({
    data: {
      id: "recovery-flow-id",
    },
  }),
  createBrowserSettingsFlow: vi.fn().mockResolvedValue({
    data: {
      id: "settings-flow-id",
    },
  }),
  getLoginFlow: vi.fn().mockResolvedValue({
    data: {
      id: "login-flow-id",
      type: "browser",
      ui: {},
    },
  }),
  updateLoginFlow: vi.fn().mockResolvedValue({
    data: {
      session: mockOrySession,
      session_token: "session-token",
    },
  }),
}));

// Mock OAuth2Api class
export const OAuth2Api = vi.fn().mockImplementation(() => ({
  oauth2TokenExchange: vi.fn().mockResolvedValue({
    data: {
      access_token: MOCK_ORY_ACCESS_TOKEN,
      token_type: "bearer",
      expires_in: 3600,
      refresh_token: "mock-refresh-token",
      scope: "openid email profile",
    },
  }),
}));
