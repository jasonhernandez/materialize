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
 */

import createClient from "openapi-fetch";

import { NOT_SUPPORTED_MESSAGE } from "~/config/AppConfig";

import { apiClient } from "./apiClient";
import {
  handleOpenApiResponse,
  handleOpenApiResponseWithBody,
} from "./openApiUtils";
import { components, paths } from "./schemas/global-api";
import { OpenApiRequestOptions } from "./types";

export type OrganizationMemberRole = components["schemas"]["OrgRole"];
export type OrganizationMember = components["schemas"]["OrganizationMember"];
export type OrganizationInvite = components["schemas"]["CreateInviteResponse"];

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

const client =
  apiClient.type === "cloud"
    ? createClient<paths>({
        baseUrl: apiClient.cloudGlobalApiBasePath,
        fetch: apiClient.cloudApiFetch,
      })
    : null;

const getClient = () => {
  if (client === null) {
    throw new Error(NOT_SUPPORTED_MESSAGE);
  }
  return client;
};

/**
 * Lists the members of the calling user's organization. Any member may
 * call this. Ory-backed organizations only.
 */
export async function listOrganizationMembers(
  requestOptions: OpenApiRequestOptions = {},
) {
  const { headers, ...options } = requestOptions;
  const { data, response } = await getClient().GET("/api/members", {
    signal: requestOptions?.signal,
    headers,
    ...options,
  });
  return handleOpenApiResponseWithBody(data, response);
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
  const { headers, ...options } = requestOptions;
  const { data, response } = await getClient().POST("/api/invites", {
    signal: requestOptions?.signal,
    headers,
    body: {
      email,
      role,
    },
    ...options,
  });
  return handleOpenApiResponseWithBody(data, response);
}

/**
 * Removes a member from the organization. Admin only; the server rejects
 * removing yourself (400) and unknown members (404).
 */
export async function removeOrganizationMember(
  memberId: string,
  requestOptions: OpenApiRequestOptions = {},
) {
  const { headers, ...options } = requestOptions;
  const { data, response } = await getClient().DELETE(
    "/api/members/{identity_id}",
    {
      params: {
        path: {
          identity_id: memberId,
        },
      },
      signal: requestOptions?.signal,
      headers,
      ...options,
    },
  );
  return handleOpenApiResponse(data, response);
}

/** Changes a member's role. Admin only. */
export async function setOrganizationMemberRole(
  memberId: string,
  role: OrganizationMemberRole,
  requestOptions: OpenApiRequestOptions = {},
) {
  const { headers, ...options } = requestOptions;
  const { data, response } = await getClient().PATCH(
    "/api/members/{identity_id}",
    {
      params: {
        path: {
          identity_id: memberId,
        },
      },
      signal: requestOptions?.signal,
      headers,
      body: {
        role,
      },
      ...options,
    },
  );
  return handleOpenApiResponse(data, response);
}
