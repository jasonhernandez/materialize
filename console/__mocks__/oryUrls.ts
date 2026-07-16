// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Node-safe stand-in for ~/config/oryUrls used by the Playwright config
 * (see the module-alias registration in playwright.config.ts). The real
 * module reads Vite's `import.meta.env`, which Playwright's CJS
 * transpilation cannot execute. E2E runs configure the equivalent values
 * through plain environment variables.
 */

export function getOryProjectUrl(_stack: string): string {
  return process.env.VITE_ORY_PROJECT_URL || "https://ory.example.invalid";
}

export function getOryOAuth2ClientId(_stack: string): string {
  return process.env.VITE_ORY_OAUTH2_CLIENT_ID || "";
}

export function getOryJwkUrl(stack: string): string {
  return `${getOryProjectUrl(stack)}/.well-known/jwks.json`;
}
