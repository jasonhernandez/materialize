// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { beforeEach, describe, expect, it, vi } from "vitest";

import { getAccessToken } from "~/api/fronteggToken";

import {
  decodeTokenIssuer,
  getSessionAuthProvider,
} from "./sessionAuthProvider";

vi.mock("~/api/fronteggToken", () => ({
  getAccessToken: vi.fn(),
}));

function fakeToken(payload: object) {
  const encode = (value: object) =>
    btoa(JSON.stringify(value))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
  return `${encode({ alg: "RS256", typ: "JWT" })}.${encode(payload)}.signature`;
}

const ORY_ISSUER = "https://slug.projects.oryapis.com";
const FRONTEGG_ISSUER = "https://admin.cloud.materialize.com";

describe("decodeTokenIssuer", () => {
  it("extracts the iss claim", () => {
    expect(decodeTokenIssuer(fakeToken({ iss: ORY_ISSUER }))).toBe(ORY_ISSUER);
  });

  it("returns undefined for malformed tokens", () => {
    expect(decodeTokenIssuer("not-a-jwt")).toBeUndefined();
    expect(decodeTokenIssuer("a.!!!.c")).toBeUndefined();
    expect(decodeTokenIssuer(fakeToken({ sub: "x" }))).toBeUndefined();
    expect(decodeTokenIssuer(fakeToken({ iss: 42 }))).toBeUndefined();
  });
});

describe("getSessionAuthProvider", () => {
  beforeEach(() => {
    vi.mocked(getAccessToken).mockReset();
  });

  it("detects Ory project issuers", () => {
    vi.mocked(getAccessToken).mockReturnValue(fakeToken({ iss: ORY_ISSUER }));
    expect(getSessionAuthProvider()).toBe("ory");
  });

  it("treats Frontegg issuers as frontegg", () => {
    vi.mocked(getAccessToken).mockReturnValue(
      fakeToken({ iss: FRONTEGG_ISSUER }),
    );
    expect(getSessionAuthProvider()).toBe("frontegg");
  });

  it("does not match oryapis.com lookalike issuers", () => {
    vi.mocked(getAccessToken).mockReturnValue(
      fakeToken({ iss: "https://evil.example/x.projects.oryapis.com" }),
    );
    expect(getSessionAuthProvider()).toBe("frontegg");
    vi.mocked(getAccessToken).mockReturnValue(
      fakeToken({ iss: "https://xprojects.oryapis.com.evil.example" }),
    );
    expect(getSessionAuthProvider()).toBe("frontegg");
  });

  it("defaults to frontegg without a token or issuer", () => {
    vi.mocked(getAccessToken).mockReturnValue(null);
    expect(getSessionAuthProvider()).toBe("frontegg");
    vi.mocked(getAccessToken).mockReturnValue("garbage");
    expect(getSessionAuthProvider()).toBe("frontegg");
  });
});
