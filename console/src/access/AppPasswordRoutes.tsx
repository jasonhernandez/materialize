// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import { useAtomValue } from "jotai";
import React from "react";
import { Route } from "react-router-dom";

import { authProviderAtom } from "~/auth/authProviderAtom";
import { User } from "~/external-library-wrappers/frontegg";
import { SentryRoutes } from "~/sentry";

import AppPasswordsPage from "./AppPasswordsPage";
import MzCliAppPasswordPage from "./MzCliAppPasswordPage";
import OryAppPasswordsPage from "./OryAppPasswordsPage";

export const AppPasswordRoutes = ({ user }: { user: User }) => {
  // Frontegg app passwords are managed via the Frontegg REST API; under
  // Ory they are Talos-backed keys managed through the sync-server.
  const authProvider = useAtomValue(authProviderAtom);
  const passwordsPage =
    authProvider === "ory" ? (
      <OryAppPasswordsPage user={user} />
    ) : (
      <AppPasswordsPage user={user} />
    );
  return (
    <SentryRoutes>
      <Route path="/" element={passwordsPage} />
      <Route path="cli" element={<MzCliAppPasswordPage user={user} />} />
    </SentryRoutes>
  );
};
