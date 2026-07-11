// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Determines which auth provider issued the current session's access
 * token, by inspecting the token's `iss` claim. During the Frontegg → Ory
 * migration (cloud/doc/design/20260710_ory_auth_migration.md) both
 * providers' tokens are valid; UI that differs per provider (e.g. app
 * password management) branches on this.
 */

import { getAccessToken } from "~/api/fronteggToken";

export type SessionAuthProvider = "frontegg" | "ory";

/** Ory Network project issuers look like https://{slug}.projects.oryapis.com. */
const ORY_ISSUER_SUFFIX = ".projects.oryapis.com";

export function decodeTokenIssuer(token: string): string | undefined {
  const parts = token.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const payload = JSON.parse(
      atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    return typeof payload.iss === "string" ? payload.iss : undefined;
  } catch {
    return undefined;
  }
}

export function getSessionAuthProvider(): SessionAuthProvider {
  const token = getAccessToken();
  if (!token) return "frontegg";
  const issuer = decodeTokenIssuer(token);
  if (issuer) {
    try {
      if (new URL(issuer).hostname.endsWith(ORY_ISSUER_SUFFIX)) {
        return "ory";
      }
    } catch {
      // Not a URL-shaped issuer; fall through to frontegg.
    }
  }
  return "frontegg";
}
