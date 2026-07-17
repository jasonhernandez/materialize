// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { http, HttpResponse } from "msw";

import { OrganizationInvite, OrganizationMember } from "~/api/orgMembers";

export const buildMember = (
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember => ({
  id: "6b29fc40-ca47-1067-b31d-00dd010662da",
  email: "admin@example.com",
  role: "admin",
  joinedAt: "2026-07-01T12:00:00Z",
  ...overrides,
});

export const buildMembersResponse = (
  options: {
    members?: OrganizationMember[];
    /** Wrap the list in the suite's standard Paginated shape. */
    paginated?: boolean;
    status?: number;
  } = {},
) =>
  http.get("*/api/members", () => {
    const members = options.members ?? [buildMember()];
    const payload = options.paginated ? { data: members } : members;
    return HttpResponse.json(payload, { status: options.status ?? 200 });
  });

export const buildInviteResponse = (
  options: {
    invite?: OrganizationInvite;
    status?: number;
    onRequest?: (body: unknown) => void;
  } = {},
) =>
  http.post("*/api/invites", async ({ request }) => {
    options.onRequest?.(await request.json());
    if (options.status && options.status >= 400) {
      return HttpResponse.json(
        { message: "conflict" },
        { status: options.status },
      );
    }
    const payload: OrganizationInvite = options.invite ?? {
      inviteToken: "test-invite-token",
      expiresAt: "2026-07-23T12:00:00Z",
    };
    return HttpResponse.json(payload, { status: options.status ?? 200 });
  });

export const buildRemoveMemberResponse = (
  options: {
    status?: number;
    onRequest?: (memberId: string) => void;
  } = {},
) =>
  http.delete("*/api/members/:memberId", ({ params }) => {
    options.onRequest?.(params.memberId as string);
    const status = options.status ?? 204;
    if (status >= 400) {
      return HttpResponse.json({ message: "error" }, { status });
    }
    return new HttpResponse(null, { status });
  });

export const buildSetMemberRoleResponse = (
  options: {
    status?: number;
    onRequest?: (memberId: string, body: unknown) => void;
  } = {},
) =>
  http.patch("*/api/members/:memberId", async ({ params, request }) => {
    options.onRequest?.(params.memberId as string, await request.json());
    const status = options.status ?? 204;
    if (status >= 400) {
      return HttpResponse.json({ message: "error" }, { status });
    }
    return new HttpResponse(null, { status });
  });
