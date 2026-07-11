# Ory login and account UX for the cloud console

- Associated: cloud/doc/design/20260710_ory_auth_migration.md,
  jasonhernandez/materialize#12 (dual-provider foundations),
  jasonhernandez/materialize#13 (Ory Elements spike),
  console/doc/ory-elements-spike.md

## The Problem

Materialize Cloud is migrating identity from Frontegg to Ory. The cloud
APIs already validate both providers' JWTs, discovery
(`POST /api/auth/discovery`) routes an email to its provider, and
app-password management for Ory-backed organizations is built. What does
not exist is a way for a user in an Ory-backed organization to sign in
to the console at all. The console's login is Frontegg's embedded box,
and its session handling assumes Frontegg's SDK everywhere: the API
client reads access tokens from Frontegg's `ContextHolder`, the runtime
config derives the user object from Frontegg hooks, and logout is a
Frontegg route.

This doc covers the console side of the migration: an email-first entry
point, the Ory self-service flows (login, registration, recovery,
verification, settings, MFA), and the session abstraction that lets the
rest of the console work identically over either provider. Frontegg and
Ory organizations must coexist for the whole migration window, and
Frontegg users must see no regression.

## Success Criteria

- A user in an Ory-backed organization can sign in with email and
  password and use the console exactly as a Frontegg user can: all cloud
  API and environmentd calls authenticate, feature flags and analytics
  identify them, and logout works.
- A migrated user (Frontegg password hashes are not exportable) can set
  a new password through the recovery flow and understands why they are
  being asked to.
- Frontegg users see at most one new screen (the email-first entry) and
  no other behavior change. Any discovery failure falls back to the
  Frontegg flow.
- Ory users can manage password, TOTP, and WebAuthn from the console.
- The auth surface is testable without a live Ory project (mocked
  Kratos payloads) and verified end to end against a real one.

## Out of Scope

- pgwire/HTTP connection-time validation of Talos app passwords in
  environmentd/balancerd. Tracked in the cloud design doc as an open
  materialize-repo item.
- Migration tooling (bulk identity import, recovery-email campaigns,
  flipping the LaunchDarkly `ory-auth-enabled` flag per domain).
- Per-customer SAML self-service (enterprise-parity workstream) and
  SCIM/group sync.
- Self-managed console auth (password, SASL, generic OIDC), which is
  unchanged.
- Organization impersonation, which bypasses login entirely.
- Removing the Frontegg SDK. It stays mounted for the whole migration
  window. Shrinking boot cost and bundle size comes after cutover.

## Solution Proposal

Four pieces, buildable in order: a session abstraction with two narrow
integration points, an Ory session model based on Kratos tokenized
sessions, an email-first entry screen that routes via discovery, and
the self-service flows rendered with Ory Elements.

### Session abstraction

The audit of Frontegg coupling found two choke points rather than the
feared thirty-one:

- **Tokens.** `CloudApiClient#getAccessToken` (src/api/apiClient.ts) is
  the only token source for HTTP middleware and websocket auth, with a
  duplicate in `src/api/fronteggToken.ts` used by non-React callers.
- **Identity.** `CloudConfigElementWrapper` (src/config/AppConfigSwitch.tsx)
  builds `runtimeConfig.user` from Frontegg's `useAuthUser`, and
  everything identity-shaped downstream (LaunchDarkly context, Segment
  identify, profile UI) reads that.

We introduce a `sessionAuth` module that both choke points call instead
of Frontegg directly:

```
getSessionToken(): string | undefined   // JWT for Authorization headers
getSessionUser(): SessionUser | undefined
logout(): Promise<void>
```

Internally it dispatches on the active provider. The provider is
determined by the existing issuer sniffing in
`src/utils/sessionAuthProvider.ts` when a token is present, seeded by a
`mz-session-auth-provider` localStorage hint written at login time so
the app knows which provider to hydrate before a token exists. Any
ambiguity resolves to frontegg, preserving today's behavior exactly.

`SessionUser` is a small normalized shape (id, email, name, org id)
mapped from Frontegg's `User` or the Ory identity. Consumers that need
Frontegg-specific fields (for example the Frontegg app-password page)
keep using the Frontegg wrapper directly, gated behind
`getSessionAuthProvider()` checks as `AppPasswordRoutes` already does.

The authenticated-route gate (`CloudFronteggAuthenticatedRoutes` in
src/platform/UnauthenticatedRoutes.tsx) grows an Ory branch: if the
stored provider is ory and an Ory session is live, render the app
without consulting Frontegg's `useIsAuthenticated`.

### Ory session model

Kratos issues a session cookie on `{slug}.projects.oryapis.com`. The
cloud APIs and environmentd want a JWT carrying the organization id.
Ory Network's session tokenizer bridges the two: `GET /sessions/whoami`
with `tokenize_as=<jwt-template>` returns the session plus a short-lived
JWT whose template projects `ext.organization_id` from identity
metadata, which the cloud validators already accept.

- **Source of truth**: the Kratos session cookie, managed by the
  browser. Survives reloads, revocable server-side.
- **The JWT**: held in memory only, refreshed by re-calling whoami
  before expiry and on any 401 from our APIs. Never persisted to
  storage.
- On boot with provider hint ory: call whoami. Live session, mint the
  JWT and render the app. No session, send the user to login.

Logout dispatches on provider: Frontegg keeps its `/account/logout`
route, Ory calls `createBrowserLogoutFlow`, follows the logout URL, and
drops the in-memory JWT.

### Email-first entry

A new console-owned route, `/auth/login`, becomes the redirect target
for unauthenticated cloud users:

1. User enters their email.
2. Console calls `discoverAuthProvider(email)` (already built, answered
   by the `ory-auth-enabled` LaunchDarkly flag on an email-domain
   context, defaults frontegg, rate-limited).
3. `frontegg`: forward to `/account/login` with the original
   `redirectUrl` preserved. The Frontegg box renders exactly as today.
4. `ory`: render the Ory login flow in place.

`/account/login` remains fully functional as a direct URL, so bookmarks
and the fail-safe path (any discovery error forwards there) behave like
today. Rollout is two-staged: the screen ships dark first (reachable by
URL, redirect target unchanged), then a one-line PR flips the redirect
target once it has soaked on staging. LaunchDarkly is not available
pre-auth in the console, so the flip is a deploy rather than a flag,
and rollback is reverting one line.

Migrated users get the "your organization upgraded its sign-in"
affordance here: when discovery says ory, the login card carries a
notice with a link into the recovery flow ("first time signing in since
the upgrade? Set your new password"). Recovery is the migration onramp,
not an edge case, since Frontegg hashes cannot be exported.

### Self-service flows with Ory Elements

Per the spike (console/doc/ory-elements-spike.md), the flows render
with `@ory/elements-react` 1.2.0's default theme, re-branded to the
Materialize palette via CSS variables. The spike validated SPA
compatibility, theming, and flow lifecycle against mocked Kratos
payloads. The spike's `useOryFlowQuery` hook (create or resume by
`?flow=` param, restart on 410, cache-seed on id sync) and the
`orySpikeTheme.css` overrides graduate from `src/ory/` spike code into
the production routes under `/auth/`:

- `/auth/login`, `/auth/registration`, `/auth/recovery`,
  `/auth/verification`: unauthenticated, mounted before the Frontegg
  redirect.
- Settings (password change, TOTP, WebAuthn) mounts inside the
  authenticated app for Ory sessions, alongside where Frontegg's
  AdminPortal serves the same needs today.

Per-slot component overrides and the headless core remain available if
design wants more than the CSS-variable re-brand, but the bar is parity
with the Frontegg box, which the re-brand meets.

### Phasing

1. **M1, login.** `sessionAuth` module, tokenized-session model, both
   choke points switched over, email-first screen dark, Ory login flow.
   Exit: an Ory user on a personal stack uses the full console.
2. **M2, migration onramp.** Recovery, registration, verification
   flows, the upgraded-sign-in affordance, email round-trips verified
   live. Exit: a migrated test user sets a password and signs in.
3. **M3, account management.** Settings flow with TOTP and WebAuthn,
   redirect-to-login on Kratos `privileged_session` re-auth errors.
4. **M4, SSO.** OIDC social/enterprise login through the same flows
   (Kratos renders provider buttons as flow nodes, so Elements picks
   them up without new UI). Reuses no react-oidc-context machinery
   despite earlier assumptions, see Alternatives.
5. **Flip the redirect target** to `/auth/login` and enable the first
   real domain via the LaunchDarkly flag.

### Testing and observability

- Unit and component: the spike's MSW pattern with realistic Kratos
  fixtures extends to every flow and to the sessionAuth dispatch (both
  providers, boot hydration, 401 refresh, logout).
- The Frontegg path keeps its existing e2e coverage, which doubles as
  the no-regression check when the redirect target flips.
- E2E for Ory needs a personal cloud stack with an Ory project (manual
  creation, see the cloud design doc) and a mail-capture strategy for
  recovery and verification. Ory Network test projects support
  retrieving codes via API for automation.
- Sentry: login-flow failures and discovery fallbacks get breadcrumbs
  so silent fail-safe-to-frontegg does not mask a broken Ory path.

## Minimal Viable Prototype

Two steps, one done:

1. **Done**: the Elements spike (jasonhernandez/materialize#13), which
   de-risked rendering, theming, and flow lifecycle with mocked Kratos
   payloads.
2. **Next**: on a personal stack with a real Ory project, wire a
   tokenized whoami JWT into `apiClient` behind the spike's
   localStorage opt-in and click through the console. This proves the
   session model (cookie, JWT template with org id, refresh) before the
   sessionAuth refactor lands.

## Alternatives

**Session model: OIDC via Hydra with react-oidc-context** (as the
self-managed console does for Ory). Standard, and the console has the
machinery. Rejected for cloud because it adds an OAuth2 layer (client
registration per stack, redirect round-trips, token refresh machinery)
on top of a Kratos session we already hold, and the settings and
recovery flows need direct Kratos session cookies anyway, so Hydra
would be additive complexity rather than a replacement. The tokenizer
is Ory Network's documented pattern for exactly this shape.

**Session model: cookie-only, no JWT in the console.** Cloud APIs would
need to accept the Kratos cookie or exchange it server-side. Rejected:
the cloud APIs and environmentd HTTP/websocket paths are JWT-bearer
based, dual validation is already built and deployed around JWTs, and
cross-site cookies to `oryapis.com` from API calls would be strictly
worse for Safari.

**UI: hand-built forms on `@ory/client-fetch`.** Full design control,
no new dependency surface. Rejected per the spike: the `ui.nodes`
contract has acknowledged implicit rules (identifier-node handling,
script-node execution for WebAuthn, per-group filtering), estimated at
3 to 6 weeks to reimplement plus permanent upkeep, for a surface whose
design bar is parity with the Frontegg box.

**UI: Elements headless core with Chakra components.** Kept as an
incremental option per component slot, not a prerequisite. Doing it up
front adds roughly 30 slot components before any user value ships.

**Entry: replace `/account/login` outright instead of a new route.**
Rejected: Frontegg's SDK owns that route's rendering, direct-URL
compatibility during the window matters, and keeping it intact is the
fail-safe target.

## Open questions

- Ory JWT template and `ext.organization_id` claim setup: Terraform
  (`ory_project_config`) or manual per project. Shared question with
  the cloud design doc, should be resolved during the MVP step.
- Kratos session lifetime and JWT template TTL values, and whether
  whoami-based refresh needs proactive scheduling or 401-retry is
  enough.
- Custom auth domain (for example `auth.materialize.com`) to keep
  cookies first-party and sidestep the Safari third-party-cookie issue
  (ory/elements#369). Likely wanted before real customers, needs cloud
  infra work. Decide before M2 exits.
- Multi-organization users. Frontegg has per-tenant sessions
  (`enableSessionPerTenant`), Ory identities belong to one organization
  in the current model. Confirm no migrating customer needs org
  switching, or scope it.
- Whether Segment, Intercom, and HubSpot signup events (currently fired
  from Frontegg provider callbacks) need Ory-side equivalents in M2 or
  can wait for the sync-server webhooks to cover them.
- Elements locale trimming (the lazy chunk is ~525 kB gzip, mostly
  react-intl locale data we do not use).
