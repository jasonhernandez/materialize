// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import {
  buildInviteResponse,
  buildMember,
  buildMembersResponse,
  buildRemoveMemberResponse,
  buildSetMemberRoleResponse,
} from "~/api/mocks/orgMembersHandlers";
import server from "~/api/mocks/server";

import { OpenApiFetchError } from "./OpenApiFetchError";
import {
  buildInviteLink,
  INVITE_ACCEPT_PATH,
  inviteOrganizationMember,
  listOrganizationMembers,
  removeOrganizationMember,
  setOrganizationMemberRole,
} from "./orgMembers";

describe("orgMembers", () => {
  describe("listOrganizationMembers", () => {
    it("parses a bare-array response", async () => {
      const member = buildMember();
      server.use(buildMembersResponse({ members: [member] }));
      const { data } = await listOrganizationMembers();
      expect(data).toEqual([member]);
    });

    it("parses a Paginated response", async () => {
      const member = buildMember({ email: "someone@example.com" });
      server.use(buildMembersResponse({ members: [member], paginated: true }));
      const { data } = await listOrganizationMembers();
      expect(data).toEqual([member]);
    });

    it("throws OpenApiFetchError on failure", async () => {
      server.use(buildMembersResponse({ status: 500 }));
      await expect(listOrganizationMembers()).rejects.toThrowError(
        OpenApiFetchError,
      );
    });
  });

  describe("inviteOrganizationMember", () => {
    it("posts the email and role and returns the invite", async () => {
      let requestBody: unknown;
      server.use(
        buildInviteResponse({
          onRequest: (body) => {
            requestBody = body;
          },
        }),
      );
      const { data } = await inviteOrganizationMember({
        email: "new@example.com",
        role: "member",
      });
      expect(requestBody).toEqual({ email: "new@example.com", role: "member" });
      expect(data.inviteToken).toEqual("test-invite-token");
      expect(data.expiresAt).toEqual("2026-07-23T12:00:00Z");
    });

    it("surfaces a 409 for an existing member", async () => {
      server.use(buildInviteResponse({ status: 409 }));
      await expect(
        inviteOrganizationMember({ email: "dup@example.com", role: "member" }),
      ).rejects.toMatchObject({ status: 409 });
    });
  });

  describe("removeOrganizationMember", () => {
    it("issues a DELETE for the member", async () => {
      let removedId: string | undefined;
      server.use(
        buildRemoveMemberResponse({
          onRequest: (memberId) => {
            removedId = memberId;
          },
        }),
      );
      await removeOrganizationMember("some-identity-id");
      expect(removedId).toEqual("some-identity-id");
    });

    it("surfaces a 400 when removing yourself", async () => {
      server.use(buildRemoveMemberResponse({ status: 400 }));
      await expect(removeOrganizationMember("my-own-id")).rejects.toMatchObject(
        { status: 400 },
      );
    });
  });

  describe("setOrganizationMemberRole", () => {
    it("issues a PATCH with the new role", async () => {
      let patched: { memberId: string; body: unknown } | undefined;
      server.use(
        buildSetMemberRoleResponse({
          onRequest: (memberId, body) => {
            patched = { memberId, body };
          },
        }),
      );
      await setOrganizationMemberRole("some-identity-id", "admin");
      expect(patched).toEqual({
        memberId: "some-identity-id",
        body: { role: "admin" },
      });
    });
  });

  describe("buildInviteLink", () => {
    it("builds a console-origin link with the URL-encoded token", () => {
      const link = buildInviteLink("token/with?chars");
      expect(link).toEqual(
        `${window.location.origin}${INVITE_ACCEPT_PATH}?token=token%2Fwith%3Fchars`,
      );
    });
  });
});
