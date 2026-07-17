// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Organization-member management for Ory-backed organizations.
 *
 * Frontegg-backed organizations manage members through the embedded
 * Frontegg AdminPortal; Ory-backed organizations manage them through the
 * cloud global API. See cloud/doc/design/20260710_ory_auth_migration.md.
 *
 * These endpoints are not yet part of the generated OpenAPI schema
 * (~/api/schemas/global-api), so this module issues requests directly
 * through the cloud API fetch (which attaches the session's bearer token)
 * rather than through the typed openapi-fetch client.
 * TODO(ory-migration): move to the generated client once the endpoints
 * land in the global-api OpenAPI spec.
 */

import { NOT_SUPPORTED_MESSAGE } from "~/config/AppConfig";

import { apiClient } from "./apiClient";
import { OpenApiFetchError } from "./OpenApiFetchError";
import { OpenApiRequestOptions } from "./types";

export type OrganizationMemberRole = "admin" | "member";

export interface OrganizationMember {
  /** Ory identity UUID. */
  id: string;
  email: string;
  role: OrganizationMemberRole;
  /** ISO 8601 datetime. */
  joinedAt: string;
}

export interface OrganizationInvite {
  inviteToken: string;
  /** ISO 8601 datetime. */
  expiresAt: string;
}

/**
 * Console route that will accept an invite token.
 *
 * TODO(ory-migration): acceptance-side routing for this path does not
 * exist yet; invites created today can only be redeemed once that flow
 * ships. Keep this constant in sync with the acceptance route when it
 * lands.
 */
export const INVITE_ACCEPT_PATH = "/auth/invite";

/** Builds the shareable invite link for an invite token. */
export function buildInviteLink(inviteToken: string): string {
  return `${window.location.origin}${INVITE_ACCEPT_PATH}?token=${encodeURIComponent(
    inviteToken,
  )}`;
}

function getCloudApiClient() {
  if (apiClient.type !== "cloud") {
    throw new Error(NOT_SUPPORTED_MESSAGE);
  }
  return apiClient;
}

async function cloudApiRequest(
  path: string,
  init: RequestInit,
  requestOptions: OpenApiRequestOptions = {},
): Promise<{ response: Response; body: unknown }> {
  const client = getCloudApiClient();
  const response = await client.cloudApiFetch(
    `${client.cloudGlobalApiBasePath}${path}`,
    {
      ...requestOptions,
      ...init,
    },
  );
  let body: unknown;
  const text = await response.text();
  if (text.length > 0) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    throw new OpenApiFetchError(
      response.status,
      (body as string | object) ?? "Empty response",
    );
  }
  return { response, body };
}

/**
 * The members list may arrive either as a bare array or wrapped in the
 * suite's standard `Paginated` shape (`{ data: [...] }`); tolerate both.
 */
function parseMembersPayload(payload: unknown): OrganizationMember[] {
  if (Array.isArray(payload)) {
    return payload as OrganizationMember[];
  }
  if (
    payload !== null &&
    typeof payload === "object" &&
    Array.isArray((payload as { data?: unknown }).data)
  ) {
    return (payload as { data: OrganizationMember[] }).data;
  }
  throw new OpenApiFetchError(200, "Unexpected members response shape");
}

/**
 * Lists the members of the calling user's organization. Any member may
 * call this. Ory-backed organizations only.
 */
export async function listOrganizationMembers(
  requestOptions: OpenApiRequestOptions = {},
) {
  const { body } = await cloudApiRequest(
    "/api/members",
    { method: "GET" },
    requestOptions,
  );
  return { data: parseMembersPayload(body) };
}

/**
 * Invites a new member to the organization. Admin only. Returns an invite
 * token; no email is sent — the caller must deliver the invite link
 * (see {@link buildInviteLink}). 409 when the email is already a member.
 */
export async function inviteOrganizationMember(
  { email, role }: { email: string; role: OrganizationMemberRole },
  requestOptions: OpenApiRequestOptions = {},
) {
  const { body } = await cloudApiRequest(
    "/api/invites",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, role }),
    },
    requestOptions,
  );
  return { data: body as OrganizationInvite };
}

/**
 * Removes a member from the organization. Admin only; the server rejects
 * removing yourself (400) and unknown members (404).
 */
export async function removeOrganizationMember(
  memberId: string,
  requestOptions: OpenApiRequestOptions = {},
) {
  await cloudApiRequest(
    `/api/members/${encodeURIComponent(memberId)}`,
    { method: "DELETE" },
    requestOptions,
  );
}

/** Changes a member's role. Admin only. */
export async function setOrganizationMemberRole(
  memberId: string,
  role: OrganizationMemberRole,
  requestOptions: OpenApiRequestOptions = {},
) {
  await cloudApiRequest(
    `/api/members/${encodeURIComponent(memberId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    },
    requestOptions,
  );
}
