// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { screen, waitFor } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import React from "react";

import { renderComponent } from "~/test/utils";

import { OryInvitePage } from "./OryInvitePage";

const createBrowserRegistrationFlow = vi.fn();
const updateRegistrationFlow = vi.fn();

vi.mock("./oryConfig", () => ({
  createOryFrontendApi: () => ({
    createBrowserRegistrationFlow,
    updateRegistrationFlow,
  }),
}));

const REGISTRATION_FLOW = {
  id: "test-flow-id",
  ui: {
    action: "https://ory.example.com/self-service/registration",
    method: "POST",
    nodes: [
      {
        attributes: { name: "csrf_token", value: "test-csrf-token" },
        messages: [],
      },
    ],
    messages: [],
  },
};

/** A Kratos 400: the flow's next state, carrying UI messages. */
function flowErrorResponse(messages: string[]) {
  const body = {
    ...REGISTRATION_FLOW,
    ui: {
      ...REGISTRATION_FLOW.ui,
      messages: messages.map((text) => ({ text })),
    },
  };
  return Object.assign(new Error("Bad Request"), {
    response: new Response(JSON.stringify(body), { status: 400 }),
  });
}

async function renderInvitePage(path = "/auth/invite?token=test-invite-token") {
  return renderComponent(<OryInvitePage />, {
    initialRouterEntries: [path],
  });
}

describe("OryInvitePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createBrowserRegistrationFlow.mockResolvedValue(REGISTRATION_FLOW);
  });

  it("explains when the invite link has no token", async () => {
    await renderInvitePage("/auth/invite");

    expect(await screen.findByText("Invalid invite link")).toBeVisible();
    expect(createBrowserRegistrationFlow).not.toHaveBeenCalled();
  });

  it("renders the signup form for a valid link", async () => {
    await renderInvitePage();

    expect(
      await screen.findByText("Join your team on Materialize"),
    ).toBeVisible();
    expect(screen.getByPlaceholderText("you@company.com")).toBeVisible();
    expect(screen.getByPlaceholderText("Choose a password")).toBeVisible();
  });

  it("submits the flow with the invite token in the transient payload", async () => {
    updateRegistrationFlow.mockResolvedValue({ identity: { id: "new-id" } });
    const assign = vi.fn();
    vi.spyOn(window, "location", "get").mockReturnValue({
      ...window.location,
      assign,
    });
    const user = userEvent.setup();
    await renderInvitePage();

    await user.type(
      await screen.findByPlaceholderText("you@company.com"),
      "invitee@example.com",
    );
    await user.type(
      screen.getByPlaceholderText("Choose a password"),
      "hunter2hunter2",
    );
    await user.click(screen.getByRole("button", { name: "Accept invitation" }));

    await waitFor(() => {
      expect(updateRegistrationFlow).toHaveBeenCalledWith({
        flow: "test-flow-id",
        updateRegistrationFlowBody: {
          method: "password",
          csrf_token: "test-csrf-token",
          password: "hunter2hunter2",
          traits: { email: "invitee@example.com" },
          transient_payload: { invite_token: "test-invite-token" },
        },
      });
    });
    // Success hands the browser to the normal OAuth2 login.
    await waitFor(() => {
      expect(assign).toHaveBeenCalledWith("/?auth_provider=ory");
    });
  });

  it("surfaces the prehook's denial message", async () => {
    // The registration prehook denies with flow-level messages (wrong
    // email for the invite, consumed/expired token); they must reach the
    // user verbatim.
    updateRegistrationFlow.mockRejectedValue(
      flowErrorResponse([
        "This invitation was issued for a different email address.",
      ]),
    );
    const user = userEvent.setup();
    await renderInvitePage();

    await user.type(
      await screen.findByPlaceholderText("you@company.com"),
      "somebody-else@example.com",
    );
    await user.type(
      screen.getByPlaceholderText("Choose a password"),
      "hunter2hunter2",
    );
    await user.click(screen.getByRole("button", { name: "Accept invitation" }));

    expect(
      await screen.findByText(
        "This invitation was issued for a different email address.",
      ),
    ).toBeVisible();
  });

  it("tells an already-signed-in user to sign out first", async () => {
    updateRegistrationFlow.mockRejectedValue(
      Object.assign(new Error("Bad Request"), {
        response: new Response(
          JSON.stringify({ error: { id: "session_already_available" } }),
          { status: 400 },
        ),
      }),
    );
    const user = userEvent.setup();
    await renderInvitePage();

    await user.type(
      await screen.findByPlaceholderText("you@company.com"),
      "invitee@example.com",
    );
    await user.type(
      screen.getByPlaceholderText("Choose a password"),
      "hunter2hunter2",
    );
    await user.click(screen.getByRole("button", { name: "Accept invitation" }));

    expect(
      await screen.findByText(/already signed in/, { exact: false }),
    ).toBeVisible();
  });
});
