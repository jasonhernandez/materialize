// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Exercises the spike's flow wiring against realistic Kratos self-service
 * payloads served by MSW. This validates our side of the contract (flow
 * create/fetch/resume, gating, error handling) and that Ory Elements
 * mounts in this app's provider stack. It is not a substitute for testing
 * against a live Ory project (CORS, cookies, email round-trips).
 */

import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import React from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import server from "~/api/mocks/server";
import { renderComponent } from "~/test/utils";

import { isOrySpikeEnabled, ORY_SPIKE_SDK_URL_KEY } from "./orySpikeConfig";
import OrySpikeRoutes from "./OrySpikeRoutes";

const SDK_URL = "https://test-slug.projects.oryapis.com";

const uiNode = (
  group: string,
  attributes: Record<string, unknown>,
  label?: { id: number; text: string },
) => ({
  type: "input",
  group,
  attributes: { disabled: false, node_type: "input", ...attributes },
  messages: [],
  meta: label ? { label: { ...label, type: "info" } } : {},
});

const buildLoginFlow = (id: string) => ({
  id,
  type: "browser",
  expires_at: "2099-01-01T00:00:00Z",
  issued_at: "2026-07-10T00:00:00Z",
  request_url: `${SDK_URL}/self-service/login/browser`,
  refresh: false,
  requested_aal: "aal1",
  state: "choose_method",
  ui: {
    action: `${SDK_URL}/self-service/login?flow=${id}`,
    method: "POST",
    messages: [],
    nodes: [
      uiNode("default", {
        name: "csrf_token",
        type: "hidden",
        value: "csrf-token-value",
        required: true,
      }),
      uiNode(
        "default",
        { name: "identifier", type: "text", value: "", required: true },
        { id: 1070004, text: "E-Mail" },
      ),
      uiNode(
        "password",
        { name: "password", type: "password", required: true },
        { id: 1070001, text: "Password" },
      ),
      uiNode(
        "password",
        { name: "method", type: "submit", value: "password" },
        { id: 1010001, text: "Sign in" },
      ),
    ],
  },
});

const session = {
  id: "session-1",
  active: true,
  expires_at: "2099-01-01T00:00:00Z",
  authenticated_at: "2026-07-10T00:00:00Z",
  authenticator_assurance_level: "aal1",
  issued_at: "2026-07-10T00:00:00Z",
  identity: {
    id: "identity-1",
    schema_id: "default",
    schema_url: `${SDK_URL}/schemas/default`,
    state: "active",
    traits: { email: "user@example.com" },
  },
};

const kratosUnauthorized = () =>
  HttpResponse.json(
    {
      error: {
        code: 401,
        status: "Unauthorized",
        message: "No valid session credentials found in the request.",
      },
    },
    { status: 401 },
  );

describe("orySpikeConfig", () => {
  it("is disabled unless the localStorage opt-in is set", () => {
    expect(isOrySpikeEnabled()).toBe(false);
    window.localStorage.setItem(ORY_SPIKE_SDK_URL_KEY, SDK_URL);
    expect(isOrySpikeEnabled()).toBe(true);
  });
});

describe("OrySpikeRoutes", () => {
  beforeEach(() => {
    window.localStorage.setItem(ORY_SPIKE_SDK_URL_KEY, SDK_URL);
  });

  afterEach(() => {
    window.localStorage.removeItem(ORY_SPIKE_SDK_URL_KEY);
  });

  it("creates a browser login flow and renders the password form", async () => {
    server.use(
      http.get(`${SDK_URL}/self-service/login/browser`, () =>
        HttpResponse.json(buildLoginFlow("flow-created")),
      ),
    );

    const { container } = await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/login"],
    });

    await waitFor(() => {
      expect(
        container.querySelector('input[name="identifier"]'),
      ).toBeInTheDocument();
    });
    expect(
      container.querySelector('input[name="password"]'),
    ).toBeInTheDocument();
    expect(
      container.querySelector('input[name="csrf_token"]'),
    ).toBeInTheDocument();
    const form = container.querySelector("form");
    expect(form).toHaveAttribute(
      "action",
      `${SDK_URL}/self-service/login?flow=flow-created`,
    );
    expect(
      container.querySelector('button[type="submit"][value="password"]'),
    ).toBeInTheDocument();
  });

  it("resumes the login flow named by the flow search param", async () => {
    let requestedFlowId: string | undefined;
    server.use(
      http.get(`${SDK_URL}/self-service/login/flows`, ({ request }) => {
        requestedFlowId =
          new URL(request.url).searchParams.get("id") ?? undefined;
        return HttpResponse.json(buildLoginFlow("flow-existing"));
      }),
    );

    const { container } = await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/login?flow=flow-existing"],
    });

    await waitFor(() => {
      expect(
        container.querySelector('input[name="identifier"]'),
      ).toBeInTheDocument();
    });
    expect(requestedFlowId).toBe("flow-existing");
  });

  it("starts a fresh login flow when the flow id is expired", async () => {
    server.use(
      http.get(`${SDK_URL}/self-service/login/flows`, () =>
        HttpResponse.json(
          {
            error: {
              code: 410,
              status: "Gone",
              message: "The login flow expired.",
            },
          },
          { status: 410 },
        ),
      ),
      http.get(`${SDK_URL}/self-service/login/browser`, () =>
        HttpResponse.json(buildLoginFlow("flow-recreated")),
      ),
    );

    const { container } = await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/login?flow=flow-expired"],
    });

    await waitFor(() => {
      expect(
        container.querySelector('input[name="identifier"]'),
      ).toBeInTheDocument();
    });
  });

  it("prompts for login when settings is requested without a session", async () => {
    server.use(
      http.get(`${SDK_URL}/self-service/settings/browser`, () =>
        kratosUnauthorized(),
      ),
    );

    await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/settings"],
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Settings requires an active Ory session/),
      ).toBeInTheDocument();
    });
  });

  it("shows no-session state on the session page", async () => {
    server.use(
      http.get(`${SDK_URL}/sessions/whoami`, () => kratosUnauthorized()),
    );

    await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/"],
    });

    await waitFor(() => {
      expect(screen.getByText(/No active Ory session/)).toBeInTheDocument();
    });
  });

  it("renders the active session with a logout button", async () => {
    server.use(
      http.get(`${SDK_URL}/sessions/whoami`, () => HttpResponse.json(session)),
    );

    await renderComponent(<OrySpikeRoutes />, {
      initialRouterEntries: ["/"],
    });

    await waitFor(() => {
      expect(screen.getByText(/Active Ory session/)).toBeInTheDocument();
    });
    expect(screen.getByText(/user@example\.com/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Log out" })).toBeInTheDocument();
  });
});
