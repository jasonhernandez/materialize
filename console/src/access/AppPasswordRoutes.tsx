// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import React from "react";
import { Route } from "react-router-dom";

import { User } from "~/external-library-wrappers/frontegg";
import { SentryRoutes } from "~/sentry";
import { getSessionAuthProvider } from "~/utils/sessionAuthProvider";

import AppPasswordsPage from "./AppPasswordsPage";
import MzCliAppPasswordPage from "./MzCliAppPasswordPage";
import OryAppPasswordsPage from "./OryAppPasswordsPage";

export const AppPasswordRoutes = ({ user }: { user: User }) => {
  // Ory-backed organizations manage app passwords via Ory Talos through
  // the cloud global API; Frontegg-backed organizations via the Frontegg
  // api-tokens API. See cloud/doc/design/20260710_ory_auth_migration.md.
  const isOrySession = getSessionAuthProvider() === "ory";
  return (
    <SentryRoutes>
      <Route
        path="/"
        element={
          isOrySession ? (
            <OryAppPasswordsPage />
          ) : (
            <AppPasswordsPage user={user} />
          )
        }
      />
      <Route path="cli" element={<MzCliAppPasswordPage user={user} />} />
    </SentryRoutes>
  );
};
