// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Configuration for the Ory Elements spike (`/ory-spike/*`).
 *
 * The spike routes only mount when an Ory SDK URL is present in
 * localStorage, so production behavior is unchanged for everyone else.
 * To enable, run the following in the devtools console and reload:
 *
 *   localStorage.setItem(
 *     "mz-ory-spike-sdk-url",
 *     "https://{slug}.projects.oryapis.com",
 *   );
 */

import { Configuration, FrontendApi } from "@ory/client-fetch";
import type { OryClientConfiguration } from "@ory/elements-react";

import storageAvailable from "~/utils/storageAvailable";

export const ORY_SPIKE_SDK_URL_KEY = "mz-ory-spike-sdk-url";

export const ORY_SPIKE_BASE_PATH = "/ory-spike";

export function getOrySpikeSdkUrl(): string | undefined {
  if (!storageAvailable("localStorage")) return undefined;
  const url = window.localStorage.getItem(ORY_SPIKE_SDK_URL_KEY);
  return url ? url : undefined;
}

export function isOrySpikeEnabled(): boolean {
  return Boolean(getOrySpikeSdkUrl());
}

/** Kratos self-service API client. Session cookies require `credentials: "include"`. */
export function buildOryFrontendClient(sdkUrl: string): FrontendApi {
  return new FrontendApi(
    new Configuration({
      basePath: sdkUrl,
      credentials: "include",
    }),
  );
}

export function buildOryClientConfiguration(
  sdkUrl: string,
): OryClientConfiguration {
  const base = `${window.location.origin}${ORY_SPIKE_BASE_PATH}`;
  return {
    sdk: {
      url: sdkUrl,
      options: {
        credentials: "include",
      },
    },
    project: {
      name: "Materialize",
      default_redirect_url: base,
      login_ui_url: `${base}/login`,
      registration_ui_url: `${base}/registration`,
      recovery_ui_url: `${base}/recovery`,
      verification_ui_url: `${base}/verification`,
      settings_ui_url: `${base}/settings`,
      error_ui_url: `${base}/login`,
      recovery_enabled: true,
      registration_enabled: true,
      verification_enabled: true,
    },
  };
}
