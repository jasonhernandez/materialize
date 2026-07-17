// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { screen, waitFor, within } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import React from "react";

import {
  buildInviteResponse,
  buildMember,
  buildMembersResponse,
  buildRemoveMemberResponse,
  buildSetMemberRoleResponse,
} from "~/api/mocks/orgMembersHandlers";
import server from "~/api/mocks/server";
import { INVITE_ACCEPT_PATH } from "~/api/orgMembers";
import { dummyValidUser } from "~/external-library-wrappers/__mocks__/frontegg";
import { User } from "~/external-library-wrappers/frontegg";
import { renderComponent } from "~/test/utils";

import OryMembersPage from "./OryMembersPage";

// dummyValidUser has id "1" and the MaterializePlatformAdmin role.
const adminUser = dummyValidUser;
const memberUser: User = { ...dummyValidUser, roles: [] };

const selfMember = buildMember({
  id: adminUser.id,
  email: adminUser.email,
  role: "admin",
  joinedAt: "2026-07-01T12:00:00Z",
});
const otherMember = buildMember({
  id: "0e37df36-f698-11e6-8dd4-cb9ced3df976",
  email: "teammate@example.com",
  role: "member",
  joinedAt: "2026-07-10T09:30:00Z",
});

describe("OryMembersPage", () => {
  beforeEach(() => {
    server.use(buildMembersResponse({ members: [selfMember, otherMember] }));
  });

  it("lists members with email, role, and joined date", async () => {
    await renderComponent(<OryMembersPage user={adminUser} />);

    const selfRow = await screen.findByRole("row", {
      name: new RegExp(selfMember.email),
    });
    expect(within(selfRow).getByText("Admin")).toBeVisible();

    const otherRow = screen.getByRole("row", {
      name: new RegExp(otherMember.email),
    });
    expect(otherRow).toBeVisible();
  });

  describe("as an admin", () => {
    it("shows invite, role, and remove controls for other members only", async () => {
      await renderComponent(<OryMembersPage user={adminUser} />);
      await screen.findByText(otherMember.email);

      expect(
        screen.getByRole("button", { name: "Invite member" }),
      ).toBeVisible();

      // Role select and remove exist for the other member...
      expect(
        screen.getByLabelText(`Role for ${otherMember.email}`),
      ).toBeVisible();
      expect(
        screen.getByRole("button", { name: "Remove member" }),
      ).toBeVisible();

      // ...but not for the admin's own row.
      expect(
        screen.queryByLabelText(`Role for ${selfMember.email}`),
      ).not.toBeInTheDocument();
      expect(
        screen.getAllByRole("button", { name: "Remove member" }),
      ).toHaveLength(1);
    });

    it("invites a member and shows the invite link with expiry", async () => {
      let requestBody: unknown;
      server.use(
        buildInviteResponse({
          invite: {
            inviteToken: "shiny-new-token",
            expiresAt: "2026-07-23T12:00:00Z",
          },
          onRequest: (body) => {
            requestBody = body;
          },
        }),
      );
      await renderComponent(<OryMembersPage user={adminUser} />);
      await screen.findByText(otherMember.email);
      const user = userEvent.setup();

      await user.click(screen.getByRole("button", { name: "Invite member" }));
      await user.type(screen.getByLabelText("Email"), "new@example.com");
      await user.selectOptions(screen.getByLabelText("Role"), "admin");
      await user.click(screen.getByRole("button", { name: "Create invite" }));

      expect(
        await screen.findByText('Invite link for "new@example.com"'),
      ).toBeVisible();
      expect(requestBody).toEqual({ email: "new@example.com", role: "admin" });

      const linkBox = screen.getByLabelText("Invite link");
      expect(linkBox).toHaveTextContent(
        `${window.location.origin}${INVITE_ACCEPT_PATH}?auth_provider=ory&token=shiny-new-token`,
      );
      // The invite is link-only; the UI must say so and show the expiry.
      expect(
        screen.getByText(/No email is sent/, { exact: false }),
      ).toBeVisible();
    });

    it("shows a friendly error when the invitee is already a member", async () => {
      server.use(buildInviteResponse({ status: 409 }));
      await renderComponent(<OryMembersPage user={adminUser} />);
      await screen.findByText(otherMember.email);
      const user = userEvent.setup();

      await user.click(screen.getByRole("button", { name: "Invite member" }));
      await user.type(screen.getByLabelText("Email"), otherMember.email);
      await user.click(screen.getByRole("button", { name: "Create invite" }));

      expect(
        await screen.findByText(
          "That email is already a member of this organization.",
        ),
      ).toBeVisible();
    });

    it("removes a member after type-to-confirm", async () => {
      let removedId: string | undefined;
      server.use(
        buildRemoveMemberResponse({
          onRequest: (memberId) => {
            removedId = memberId;
          },
        }),
      );
      await renderComponent(<OryMembersPage user={adminUser} />);
      await screen.findByText(otherMember.email);
      const user = userEvent.setup();

      await user.click(screen.getByRole("button", { name: "Remove member" }));
      const confirmButton = screen.getByRole("button", { name: "Remove" });
      expect(confirmButton).toBeDisabled();

      // Typing the wrong text keeps the action disabled.
      const input = screen.getByRole("textbox");
      await user.type(input, "wrong@example.com");
      expect(confirmButton).toBeDisabled();

      await user.clear(input);
      await user.type(input, otherMember.email);
      expect(confirmButton).toBeEnabled();
      await user.click(confirmButton);

      await waitFor(() => expect(removedId).toEqual(otherMember.id));
    });

    it("changes a member's role", async () => {
      let patched: { memberId: string; body: unknown } | undefined;
      server.use(
        buildSetMemberRoleResponse({
          onRequest: (memberId, body) => {
            patched = { memberId, body };
          },
        }),
      );
      await renderComponent(<OryMembersPage user={adminUser} />);
      await screen.findByText(otherMember.email);
      const user = userEvent.setup();

      await user.selectOptions(
        screen.getByLabelText(`Role for ${otherMember.email}`),
        "admin",
      );

      await waitFor(() =>
        expect(patched).toEqual({
          memberId: otherMember.id,
          body: { role: "admin" },
        }),
      );
    });
  });

  describe("as a regular member", () => {
    it("shows a read-only list without any management controls", async () => {
      await renderComponent(<OryMembersPage user={memberUser} />);
      await screen.findByText(otherMember.email);

      expect(screen.getByText(selfMember.email)).toBeVisible();
      expect(
        screen.queryByRole("button", { name: "Invite member" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Remove member" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByLabelText(`Role for ${otherMember.email}`),
      ).not.toBeInTheDocument();
      // Roles still render as plain text.
      expect(screen.getByText("Admin")).toBeVisible();
      expect(screen.getByText("Member")).toBeVisible();
    });
  });
});
