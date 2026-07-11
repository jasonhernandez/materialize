# Ory Elements spike

Evaluates [Ory Elements](https://github.com/ory/elements)
(`@ory/elements-react`) as the UI layer for the Ory login and account UX
in the Frontegg → Ory migration
(`cloud/doc/design/20260710_ory_auth_migration.md`). The question: do we
build the Kratos self-service flows (login, registration, recovery,
verification, settings/MFA) on Elements, on Elements' headless core with
our own Chakra components, or by hand against the Kratos `ui.nodes`
contract?

## What the spike is

Routes under `/ory-spike/*` (cloud mode only) rendering all five
self-service flows with Elements' default theme, re-branded to the
Materialize purple scale and Inter via CSS variables
(`src/ory/orySpikeTheme.css`), plus a session viewer with logout. The
flow pages fetch or create browser flows with `@ory/client-fetch` and
hand the flow object to Elements' flow components.

The spike is inert unless opted into. To try it against an Ory project:

1. Add the console origin to the Ory project's allowed CORS origins
   (Ory Network caps origins at 50 and forbids `*` and `localhost`; use
   `ory tunnel` for local dev if needed).
2. In the devtools console:
   `localStorage.setItem("mz-ory-spike-sdk-url", "https://{slug}.projects.oryapis.com")`
   and reload.
3. Visit `/ory-spike/login`.

Without the localStorage key the routes do not mount and cloud auth is
byte-for-byte the status quo.

## Package facts (as of 2026-07-10)

- `@ory/elements-react` 1.2.0 (2026-06-08) is the maintained library;
  `@ory/elements` 0.x is in explicit maintenance mode. Peer deps are only
  react/react-dom 18/19. No Next.js requirement in the package, but all
  official examples and docs are Next.js (React-only examples are an open
  request, ory/elements#598).
- We pin `@ory/client-fetch` 1.22.22 to match the version Elements pins,
  avoiding duplicate-module type skew.
- Elements' `orySdkUrl()`/`guessPotentiallyProxiedOrySdkUrl()` helpers
  read Next/Vercel env conventions. We ignore them and construct our own
  `Configuration({ basePath, credentials: "include" })`.
- Flow components cover login, registration, recovery (code and link),
  verification, settings (password, profile, TOTP, WebAuthn, passkeys,
  recovery codes, SSO linking), OAuth2 consent, and error. MFA step-up
  renders through the same login component.

## Findings

### Theming

- The default theme ships as precompiled CSS (no Tailwind toolchain
  needed) inside `@layer ory-elements`, keyed off ~100 CSS custom
  properties. It does not leak styles into the rest of the console, and
  un-layered overrides win the cascade without specificity fights.
- All variable indirection lives on `:root`/`:host`, so brand overrides
  must target `:root` (custom-property substitution happens at the
  definition site; a wrapper class does not propagate).
- Re-branding to Materialize purple + Inter took a 12-line CSS file. The
  result is not Chakra-fidelity (Ory's card anatomy, spacing, and focus
  styles remain), but the bar for this surface is "equivalent or better
  than the Frontegg login box", and it clears that bar.
- No built-in dark mode (ory/elements#560). The console has dark mode;
  the login surface today (Frontegg) is light-only, so parity again.
- Beyond CSS variables, every flow component takes a `components` prop
  overriding any slot (buttons, inputs, card chrome, section layout), so
  fidelity can be raised incrementally per-slot later without changing
  the architecture. The fully headless core entry (`@ory/elements-react`
  without `/theme`) is the end state if we ever want every pixel Chakra,
  at the cost of supplying ~30 slot components.

### SPA integration

- Works in the Vite SPA. `"use client"` directives are inert, submission
  is client-side fetch, and the bundle builds. The spike lazy-loads the
  Elements chunk so the main bundle is unaffected. That chunk is ~525 kB
  gzipped, dominated by react-intl locale data; production adoption
  should trim locales (we ship English-only today).
- We own flow lifecycle: create/fetch by `?flow=` param, restart on
  expiry. This is ~40 lines of react-query (`useOryFlowQuery`) and slots
  into the console's existing patterns. One subtlety the tests caught:
  after creating a flow and writing its id to the URL, the cache must be
  seeded under the new id or the key change refetches the flow just
  received.
- `OrySpikeRoutes.test.tsx` exercises the wiring against realistic Kratos
  payloads served by MSW: flow creation renders the full password form
  (identifier, password, csrf, submit action), `?flow=` resume fetches by
  id, expired flows (410) restart cleanly, settings without a session
  shows the sign-in prompt, and the session page handles both 401 and an
  active session. This validates our side of the contract and that
  Elements mounts in the app's provider stack; it does not cover CORS,
  cookies, or email round-trips, which still need a live project.
- Elements' ESM build uses extension-less internal imports
  (ory/elements#573). Vite resolves them, but vitest externalizes the
  package to node's stricter ESM loader, so `vitest.config.ts` inlines
  `@ory/elements-react`.
- Cross-flow links inside Elements ("Forgot password?", "Sign up") are
  plain `<a>` full-page navigations, not react-router transitions
  (ory/elements#590). Acceptable for auth pages; fixable later via the
  `Node.Anchor` component override.
- Success redirects go through `window.location.assign` to the
  `default_redirect_url`/`return_to`. Fine for auth, where a full reload
  into the app is what we want anyway.

### Risks / open items

- Slow stable release cadence (three stables in 12 months) and slow issue
  triage; a small maintainer team. Mitigation: everything we use is also
  Apache-2.0 source we can override per-slot or vendor.
- Ory session persistence is a Kratos session cookie on
  `{slug}.projects.oryapis.com`. How that hands off to an org-scoped JWT
  for the cloud APIs (the `sessionAuthProvider` abstraction) is the M1
  session-abstraction work, out of scope for this spike.
- Registration marks optional fields required (ory/elements#574) and
  translation overrides have rough edges (ory/elements#601). Paper cuts,
  not blockers.
- Safari cross-site cookie issue (ory/elements#369) only bites cross-site
  setups. A custom Ory domain (auth.materialize.com style) sidesteps it
  and is likely where production lands anyway.

## Recommendation

Use Ory Elements with the default theme plus CSS-variable branding for
M1–M3, overriding individual component slots only where the design
demands it. Hand-building on the raw `ui.nodes` contract is a 3-6 week
trap with permanent upkeep (implicit contract gotchas are acknowledged by
Ory in ory/kratos#4152: identifier-node stealing, script-node execution
for WebAuthn/TOTP, per-group filtering). The headless-core-with-Chakra
path stays available as an incremental follow-up, not a prerequisite.
