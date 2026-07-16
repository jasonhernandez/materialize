/**
 * Adapts an Ory session to the Frontegg `User` shape.
 *
 * Nearly every consumer of `runtimeConfig.user` (Tutorial, LaunchDarkly
 * context, billing pages, NavBar, ...) reads Frontegg `User` fields. Rather
 * than teaching ~20 call sites about a second provider, the Ory runtime
 * config carries a `User`-shaped view of the Ory session so both providers
 * satisfy the same contract. Fields no consumer reads are stubbed.
 */

import {
  extractOryEmail,
  extractOryName,
  extractOryPermissions,
  extractOryRoles,
  extractOryTenantId,
} from "~/api/auth";
import { getOryAccessToken } from "~/api/oryToken";
import type {
  ITeamUserPermission,
  User,
} from "~/external-library-wrappers/frontegg";
import type { OrySession } from "~/external-library-wrappers/ory";

/**
 * Permissions assumed for Ory users when the session doesn't expose any.
 *
 * Kratos never exposes `metadata_admin` (where roles/permissions live) to
 * the browser — only the admin API can read it — so client-side permission
 * checks usually cannot see the real grants. During the PoC every Ory user
 * is the founding admin of their organization (the registration prehook in
 * cloud src/sync/src/ory/prehooks.rs stamps the same role/permission set
 * into the identity's metadata_admin), so the UI assumes that founding-
 * admin set. This only gates UI affordances; the backend enforces the real
 * permissions from the identity's metadata server-side.
 *
 * TODO(ory-migration): once orgs have non-admin members, surface real
 * roles/permissions in JWT claims via a Hydra token hook and drop this.
 */
const ORY_FALLBACK_PERMISSIONS = [
  "materialize.environment.read",
  "materialize.environment.write",
  "materialize.invoice.read",
  "fe.secure.read.tenantApiTokens",
  "fe.secure.write.tenantApiTokens",
  "fe.secure.delete.tenantApiTokens",
];

const ORY_FALLBACK_ROLES = ["MaterializePlatformAdmin"];

export function orySessionToUser(session: OrySession): User {
  const identityId = session.identity?.id ?? session.id;
  const email = extractOryEmail(session) ?? "";
  const name = extractOryName(session) ?? email;
  const tenantId = extractOryTenantId(session) ?? "";
  const sessionRoles = extractOryRoles(session);
  const roles = sessionRoles.length > 0 ? sessionRoles : ORY_FALLBACK_ROLES;
  const sessionPermissions = extractOryPermissions(session);
  const permissions =
    sessionPermissions.length > 0
      ? sessionPermissions
      : ORY_FALLBACK_PERMISSIONS;

  const user: Partial<User> = {
    id: identityId,
    sub: identityId,
    email,
    name,
    profilePictureUrl: null,
    tenantId,
    tenantIds: tenantId ? [tenantId] : [],
    tenants: [],
    verified: true,
    mfaEnrolled: false,
    metadata: JSON.stringify({}),
    permissions: permissions.map((key) => ({
      key,
    })) as ITeamUserPermission[],
    roles: roles.map((key) => ({
      id: "",
      key,
      name: key,
      isDefault: false,
      permissions: [],
      vendorId: "",
      createdAt: new Date(0),
      updatedAt: new Date(0),
    })),
    accessToken: getOryAccessToken() ?? "",
    expiresIn: 0,
    expires: "",
    exp: 0,
  };

  return user as User;
}
