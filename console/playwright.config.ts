// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { defineConfig, devices, PlaywrightTestConfig } from "@playwright/test";
import { addAliases } from "module-alias";

import { E2EAuthProvider } from "./e2e-tests/util";

// HACK (SangJunBak):
// This is a workaround to allow Playwright to resolve the importAppConfig file to the stub implementation in /e2e-tests/importAppConfig.ts.
// This is necessary because Playwright requires CommonJS (which doesn't support top-level await), while our main appConfig uses ESM with top-level await.
// We use module-alias rather than tsconfig.json's compilerOptions.paths because I couldn't get it to resolve correctly.
addAliases({
  "~/config/importAppConfig": "__mocks__/importAppConfig",
});

/**
 * Auth provider test configuration.
 *
 * By default, tests run with both "frontegg" (forced) and "auto" (detection logic) modes.
 * Use E2E_AUTH_PROVIDER to run only specific auth configurations:
 *
 *   yarn test:e2e                           # Both frontegg + auto
 *   E2E_AUTH_PROVIDER=frontegg yarn test:e2e  # Only forced Frontegg
 *   E2E_AUTH_PROVIDER=ory yarn test:e2e     # Only Ory (when available)
 *   E2E_AUTH_PROVIDER=auto yarn test:e2e    # Only detection logic
 *   E2E_AUTH_PROVIDER=all yarn test:e2e     # All modes (frontegg + auto + ory when ready)
 */
const envAuthProvider = process.env.E2E_AUTH_PROVIDER as
  | E2EAuthProvider
  | "all"
  | undefined;

// Determine which auth providers to test
// Default: frontegg + auto (skip ory until backend is ready)
const authProvidersToTest: E2EAuthProvider[] = (() => {
  if (!envAuthProvider || envAuthProvider === "all") {
    // Default or "all": run frontegg and auto (add "ory" here when ready)
    return ["frontegg", "auto"];
  }
  // Single provider specified
  return [envAuthProvider as E2EAuthProvider];
})();

// Base browser configs
const chromiumBase = { ...devices["Desktop Chrome"] };
const webkitBase = { ...devices["Desktop Safari"] };

// Build project configs for each auth provider
// The authProvider property is accessed via test.info().project.use.authProvider in tests
const browserProjects = authProvidersToTest.flatMap((provider) => {
  const suffix = provider === "frontegg" ? "" : `-${provider}`;
  return [
    {
      name: `chromium${suffix}`,
      use: { ...chromiumBase, authProvider: provider },
      testIgnore: ["**/scalability/**"],
    },
    {
      name: `webkit${suffix}`,
      use: { ...webkitBase, authProvider: provider },
      testIgnore: ["**/scalability/**"],
    },
  ];
});

// Get all browser project names for teardown dependencies
const browserProjectNames = browserProjects.map((p) => p.name);

const config: PlaywrightTestConfig = defineConfig({
  projects: [
    ...browserProjects,
    {
      name: "scalability",
      use: { ...chromiumBase, authProvider: "frontegg" },
      testMatch: "**/scalability/**/*.spec.ts",
    },
    // Teardown - cleans up after all tests. We need all our tests to be
    // listed here, otherwise regions might be deleted out from under other
    // tests while they are running.
    {
      name: "teardown",
      testMatch: /global-teardown\.ts/,
      dependencies: browserProjectNames,
    },
  ],
  testDir: "e2e-tests",
  // Per test timeout
  timeout: 30 * 1000, // 30 seconds
  use: {
    acceptDownloads: true,
    // Actions such as clicks, also waitForSelector calls
    actionTimeout: 5 * 1000, // 5 seconds
    trace: "on",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    // In kind, we use self-signed certs
    ignoreHTTPSErrors: true,
  },
  // If you change this, also update the value of `NUM_PLAYWRIGHT_WORKERS` in `./e2e-tests/util.ts`
  workers: 5,
  fullyParallel: true,
});

export default config;
