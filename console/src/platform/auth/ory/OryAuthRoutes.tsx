/**
 * Ory Authentication Routes
 *
 * Defines routes for all Ory authentication flows.
 * These routes are accessible both before and after authentication,
 * depending on the specific flow.
 */

import React from "react";
import { Route, Routes } from "react-router-dom";

import { OryCallback } from "./OryCallback";
import { OryLoginPage } from "./OryLoginPage";
import { OryRecoveryPage } from "./OryRecoveryPage";
import { OryRegistrationPage } from "./OryRegistrationPage";
import { OrySettingsPage } from "./OrySettingsPage";
import { OryVerificationPage } from "./OryVerificationPage";

/**
 * Routes for Ory authentication flows.
 *
 * These routes handle:
 * - /auth/ory/callback - OAuth2 authorization code callback
 * - /auth/ory/login - Login flow
 * - /auth/ory/registration - Registration flow
 * - /auth/ory/recovery - Password recovery flow
 * - /auth/ory/verification - Email verification flow
 * - /auth/ory/settings - Account settings (requires authentication)
 */
export const OryAuthRoutes = () => {
  return (
    <Routes>
      <Route path="callback" element={<OryCallback />} />
      <Route path="login" element={<OryLoginPage />} />
      <Route path="registration" element={<OryRegistrationPage />} />
      <Route path="recovery" element={<OryRecoveryPage />} />
      <Route path="verification" element={<OryVerificationPage />} />
      <Route path="settings" element={<OrySettingsPage />} />
    </Routes>
  );
};

export default OryAuthRoutes;
