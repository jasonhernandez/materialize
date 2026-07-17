// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import React from "react";
import { Navigate, Route } from "react-router-dom";

import { User } from "~/external-library-wrappers/frontegg";
import { SentryRoutes } from "~/sentry";
import { getSessionAuthProvider } from "~/utils/sessionAuthProvider";

import OryMembersPage from "./OryMembersPage";

export const MembersRoutes = ({ user }: { user: User }) => {
  // Ory-backed organizations manage members through this console-native
  // page (cloud global API); Frontegg-backed organizations manage them in
  // the embedded Frontegg AdminPortal (Account settings → Users), so they
  // are sent home. See cloud/doc/design/20260710_ory_auth_migration.md.
  const isOrySession = getSessionAuthProvider() === "ory";
  return (
    <SentryRoutes>
      <Route
        path="/"
        element={
          isOrySession ? (
            <OryMembersPage user={user} />
          ) : (
            <Navigate to="/" replace />
          )
        }
      />
    </SentryRoutes>
  );
};
