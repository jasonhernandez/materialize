// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

import assert from "node:assert";

// We can ignore the no-restricted-imports rule here because we don't want to export these test utilities
// in our wrapper and don't need to mock these in our e2e tests either.
// eslint-disable-next-line no-restricted-imports
import { FronteggAuthenticator, HttpClient } from "@frontegg/client";
import { APIRequestContext, expect, Page, test } from "@playwright/test";
import retry, { AbortError } from "p-retry";

import { Region } from "~/api/cloudGlobalApi";
import { buildFronteggUrl } from "~/api/frontegg/index";
import { type AuthProviderType } from "~/auth/detectAuthProvider";
import { getCloudGlobalApiUrl } from "~/config/apiUrls";
import { appConfig } from "~/config/AppConfig";

/**
 * E2E auth provider options:
 * - "frontegg": Force Frontegg authentication
 * - "ory": Force Ory authentication
 * - "auto": Don't force a provider, let detection logic run
 */
export type E2EAuthProvider = AuthProviderType | "auto";

/**
 * Default auth provider for E2E tests (used when not running in a Playwright project context).
 * Prefer using getProjectAuthProvider() inside tests to get the project-specific provider.
 */
export const E2E_AUTH_PROVIDER: E2EAuthProvider =
  (process.env.E2E_AUTH_PROVIDER as E2EAuthProvider) || "frontegg";

/**
 * Get the auth provider for the current test project.
 * This reads from the project's `use.authProvider` config set in playwright.config.ts.
 * Falls back to E2E_AUTH_PROVIDER env var if not in a project context.
 *
 * Must be called from within a test or test hook (where test.info() is available).
 */
export function getProjectAuthProvider(): E2EAuthProvider {
  try {
    const projectUse = test.info().project.use as {
      authProvider?: E2EAuthProvider;
    };
    if (projectUse?.authProvider) {
      return projectUse.authProvider;
    }
  } catch {
    // test.info() not available outside test context
  }
  return E2E_AUTH_PROVIDER;
}

/**
 * Appends auth_provider parameter to a URL to force a specific auth provider.
 * If provider is "auto", returns the URL unchanged to test detection logic.
 */
export function withAuthProvider(
  url: string,
  provider: E2EAuthProvider = E2E_AUTH_PROVIDER,
): string {
  // "auto" mode: don't add auth_provider param, let detection run
  if (provider === "auto") {
    return url;
  }

  const urlObj = new URL(url, "http://localhost");
  urlObj.searchParams.set("auth_provider", provider);
  // Return just the path + search if it was a relative URL
  if (!url.startsWith("http")) {
    return urlObj.pathname + urlObj.search;
  }
  return urlObj.toString();
}

function getEnvVarOrFail(varName: string, errorMessage: string): string {
  const value = process.env[varName];
  if (!value) {
    throw new Error(errorMessage);
  }
  return value;
}

export const CONSOLE_ADDR = `${appConfig.consoleUrl.protocol}//${appConfig.consoleUrl.hostname}${
  appConfig.consoleUrl.port ? ":" + appConfig.consoleUrl.port : ""
}`;

/**
 * Stack override for runs against a personal stack (e.g.
 * E2E_STACK=jason2.dev). The node-side stack detection only understands
 * local/staging/production, and the browser needs the stack persisted
 * (mz-current-stack) before the app boots — TestContext.start handles
 * both when this is set.
 */
export const E2E_STACK = process.env.E2E_STACK;

function buildRegions(stack: string): Region[] {
  if (stack === "local") {
    return [
      {
        id: "local/kind",
        cloudProvider: "local",
        name: "kind",
        // When pointing at a local kind cluster, the region-api server lives on port 32001
        url: "http://127.0.0.1:32001",
      },
    ];
  }
  const stackString = stack === "production" ? "" : `.${stack}`;
  return [
    {
      id: "aws/us-east-1",
      cloudProvider: "aws",
      name: "us-east-1",
      url: `https://api.us-east-1.aws${stackString}.cloud.materialize.com`,
    },
    {
      id: "aws/eu-west-1",
      cloudProvider: "aws",
      name: "eu-west-1",
      url: `https://api.eu-west-1.aws${stackString}.cloud.materialize.com`,
    },
  ];
}

// If you change this, also update the value of `config.workers` in `playwright.config.ts`
export const NUM_PLAYWRIGHT_WORKERS = 5;
// currentStack is only null in flexible deployment mode and our e2e tests
// only run against mz cloud deployments.
export const STACK =
  E2E_STACK ?? (appConfig.mode === "cloud" ? appConfig.currentStack : "local");
export const IS_LOCAL_STACK = STACK === "local";
/**
 * Regions to exercise. Personal stacks typically deploy a single region,
 * so they default to us-east-1; override with
 * E2E_REGIONS="aws/us-east-1,aws/eu-west-1".
 */
const REGION_FILTER = process.env.E2E_REGIONS
  ? process.env.E2E_REGIONS.split(",")
  : ["local", "staging", "production"].includes(STACK)
    ? null
    : ["aws/us-east-1"];
export const REGIONS = buildRegions(STACK).filter(
  (r) => !REGION_FILTER || REGION_FILTER.includes(r.id),
);
export const CLOUD_GLOBAL_API_URL = getCloudGlobalApiUrl({
  stack: STACK,
  isImpersonation: false,
});
const e2eTenantStack = STACK === "local" ? "staging" : STACK;

/**
 * Lazily read so runs that only exercise the Ory provider don't require
 * the Frontegg test password.
 */
function fronteggPassword(): string {
  return getEnvVarOrFail(
    "E2E_TEST_PASSWORD",
    `Please set $E2E_TEST_PASSWORD on the environment; from the cloud repo, use 'pulumi stack output --stack materialize/${e2eTenantStack} --show-secrets console_e2e_test_password' to retrieve the value.`,
  );
}

/**
 * Credentials for the Ory auth-provider projects. There is no shared Ory
 * test-tenant pool yet: runs point at a single pre-provisioned identity
 * (for personal stacks see HANDOFF-ory-poc.md in the cloud repo).
 */
function oryCredentials(): { email: string; password: string } {
  return {
    email: getEnvVarOrFail(
      "E2E_ORY_EMAIL",
      "Please set $E2E_ORY_EMAIL to the Ory test identity's email.",
    ),
    password: getEnvVarOrFail(
      "E2E_ORY_PASSWORD",
      "Please set $E2E_ORY_PASSWORD to the Ory test identity's password.",
    ),
  };
}

/**
 * To prevent tests from trampling over each other, we need to ensure they
 * receive a unique Frontegg tenant.
 *
 * We accomplish this by figuring out how many accounts we have to work with,
 * bucketing them based on our Playwright concurrency settings, and then
 * assigning offsets into the collection based on a combination GitHub Actions
 * Run ID and the Playwright worker index.
 */
function getE2EIndex(stack: string): number {
  const totalWorkers = NUM_PLAYWRIGHT_WORKERS ?? 1;
  const totalAccounts = stack === "staging" ? 50 : 5;
  const workerOffset = parseInt(process.env.TEST_PARALLEL_INDEX ?? "0");
  const runId = parseInt(process.env.GITHUB_RUN_ID ?? "0");
  const totalE2EGroups = Math.floor(totalAccounts / totalWorkers);
  const e2eStartOffset = runId % totalE2EGroups;
  return e2eStartOffset + workerOffset;
}

const FRONTEGG_EMAIL = `infra+cloud-integration-tests-${e2eTenantStack}-console-${getE2EIndex(
  e2eTenantStack,
)}@materialize.io`;

/**
 * The email the tests sign in with. Ory-only runs use the single Ory test
 * identity; everything else uses the Frontegg test-tenant pool.
 */
export const EMAIL =
  E2E_AUTH_PROVIDER === "ory" && process.env.E2E_ORY_EMAIL
    ? process.env.E2E_ORY_EMAIL
    : FRONTEGG_EMAIL;

export const STATE_NAME = `e2e-tests/state-${process.env.TEST_PARALLEL_INDEX}.json`;

export const FRONTEGG_CLIENT_ID = process.env["E2E_FRONTEGG_CLIENT_ID"];

export const FRONTEGG_SECRET_KEY = process.env["E2E_FRONTEGG_SECRET_KEY"];

// No idea why this is so slow sometimes
export const FRONTEGG_LOADING_TIMEOUT = 60_000;

interface FronteggAuthResponse {
  /** Short-lived access token. */
  accessToken: string;
  /** Longer-lived refresh token, usable only once. */
  refreshToken: string;
  /** Time after which the access token has expired. */
  expires: string;
  /** Seconds until expiration */
  expiresIn: number;
}

export type Options = Parameters<APIRequestContext["fetch"]>[1];

function jwtPayload(token: string): Record<string, any> {
  const parts = token.split(".");
  assert(parts.length === 3, "expected a JWT");
  return JSON.parse(Buffer.from(parts[1], "base64").toString("utf8"));
}

function isJwtExpired(token: string): boolean {
  try {
    const { exp } = jwtPayload(token);
    if (!exp) return false;
    // 30s buffer for clock skew.
    return Date.now() >= exp * 1000 - 30_000;
  } catch {
    return true;
  }
}

/** A refresh deadline halfway to the token's expiry. */
function jwtHalfLife(token: string): Date {
  const { exp } = jwtPayload(token);
  const now = Date.now();
  const expiresMs = exp ? exp * 1000 : now + 60 * 60 * 1000;
  return new Date(now + Math.max(0, (expiresMs - now) / 2));
}

/** Manages an end-to-end test against Materialize Console. */
export class TestContext {
  page: Page;
  request: APIRequestContext;
  accessToken: string;
  refreshToken: string;
  refreshDeadline: Date;
  private fronteggClient: HttpClient | undefined = undefined;

  public get fronteggAPIEnabled(): boolean {
    return this.fronteggClient !== undefined;
  }

  constructor(page: Page, request: APIRequestContext) {
    this.page = page;
    this.request = request;
    this.accessToken = "";
    this.refreshToken = "";
    this.refreshDeadline = new Date(0);
    if (FRONTEGG_CLIENT_ID && FRONTEGG_SECRET_KEY) {
      const authenticator = new FronteggAuthenticator();
      authenticator.init(FRONTEGG_CLIENT_ID, FRONTEGG_SECRET_KEY);
      this.fronteggClient = new HttpClient(authenticator, {
        baseURL: "https://api.frontegg.com",
      });
    } else {
      console.info(
        "No Frontegg API credentials found. Not initializing admin API client.",
      );
    }
  }

  /** The auth provider for the current Playwright project. */
  get authProvider(): E2EAuthProvider {
    return getProjectAuthProvider();
  }

  /** Whether this test runs against the Ory auth provider. */
  get isOry(): boolean {
    return this.authProvider === "ory";
  }

  /** Start a new test. */
  static async start(page: Page, request: APIRequestContext) {
    const context = new TestContext(page, request);
    console.info("EMAIL=", EMAIL);

    if (E2E_STACK) {
      // Personal-stack runs: the console picks its stack from
      // localStorage, which must exist before the app boots.
      await page.addInitScript((stack) => {
        window.localStorage.setItem("mz-current-stack", stack);
      }, E2E_STACK);
    }

    // Provide a clean slate for the test.
    if (context.isOry) {
      // Ory has no resource-owner API grant: authenticate through the UI
      // first so API calls (disableAllRegions) can reuse the browser
      // session's token.
      await context.goto(CONSOLE_ADDR);
      await context.disableAllRegions();
      await context.goto(CONSOLE_ADDR);
    } else {
      if (context.fronteggAPIEnabled) {
        await context.setFronteggTenantBlockedStatus(false);
      }
      await context.disableAllRegions();

      // Navigate to the home page && wait for that to load.
      await context.goto(CONSOLE_ADDR);
    }
    // We assume the first page is the onboarding survey
    await page.waitForSelector("[data-testid=onboarding-survey]");
    return context;
  }

  async signIn() {
    if (this.isOry) {
      await this.signInOry();
      return;
    }
    await this.page.waitForSelector("[data-test-id=input-identifier]", {
      timeout: FRONTEGG_LOADING_TIMEOUT,
    });
    await this.page.fill("[name=identifier]", EMAIL);
    await this.page.press("[name=identifier]", "Enter");
    await this.page.waitForSelector("[name=password]"); // wait for animation
    await this.page.fill("[name=password]", fronteggPassword());
    this.page.press("[name=password]", "Enter");
    await this.waitForFronteggToLoad();
    await expect(
      this.page.getByRole("link", { name: "Logo Materialize" }),
    ).toBeVisible();
    await this.page.context().storageState({ path: STATE_NAME });
  }

  /**
   * Sign in through the Ory hosted login (identifier-first, two-step).
   * Assumes the page has already been redirected to the hosted flow.
   */
  async signInOry() {
    const { email, password } = oryCredentials();
    await this.page.waitForURL(/oryapis\.com/, {
      timeout: FRONTEGG_LOADING_TIMEOUT,
    });
    await this.page.waitForSelector(
      'input[name="identifier"], input[name="password"]',
      { state: "attached", timeout: FRONTEGG_LOADING_TIMEOUT },
    );
    // Returning-user flows carry the identifier as a hidden prefilled
    // input and go straight to the password step.
    const identifier = this.page.locator('input[name="identifier"]').first();
    if (await identifier.isVisible().catch(() => false)) {
      await identifier.fill(email);
      await this.page.getByRole("button", { name: /continue/i }).click();
    }
    await this.page.waitForSelector('input[name="password"]');
    await this.page.fill('input[name="password"]', password);
    await this.page
      .getByRole("button", { name: /sign in|continue|log in/i })
      .first()
      .click();
    // The OAuth callback lands back on the console.
    await this.waitForFronteggToLoad();
    assert(
      await this.captureOryTokenFromPage(),
      "Ory sign-in did not produce an access token",
    );
    await this.page.context().storageState({ path: STATE_NAME });
  }

  /**
   * Reads the console's Ory access token off the page, if it has a live
   * one, and adopts it for API requests.
   */
  private async captureOryTokenFromPage(): Promise<string | null> {
    const token = await this.page
      .evaluate(() => window.sessionStorage.getItem("ory_access_token"))
      .catch(() => null);
    if (!token || isJwtExpired(token)) {
      return null;
    }
    this.accessToken = token;
    this.refreshDeadline = jwtHalfLife(token);
    return token;
  }

  async ensureAuthenticated() {
    if (new Date().getTime() < this.refreshDeadline.getTime()) {
      return;
    }

    if (this.isOry) {
      // Ory has no password grant; reuse the browser session's token,
      // driving the UI sign-in if the page doesn't hold a live one.
      if (await this.captureOryTokenFromPage()) {
        return;
      }
      await this.goto(CONSOLE_ADDR);
      assert(
        await this.captureOryTokenFromPage(),
        "Ory sign-in did not produce an access token",
      );
      return;
    }

    const authUrl = buildFronteggUrl("/identity/resources/auth/v1/user");
    const response = await retry(
      () =>
        this.request.post(authUrl, {
          data: {
            email: EMAIL,
            password: fronteggPassword(),
          },
          timeout: 10 * 1000,
        }),
      // Sometimes we get ECONNRESET or requests hang on these frontegg calls
      { retries: 4 },
    );

    const text = await response.text();
    let auth: FronteggAuthResponse;
    try {
      auth = JSON.parse(text);
    } catch (e: unknown) {
      console.error(`Invalid json from ${authUrl}:\n${text}`);
      throw e as SyntaxError;
    }

    this.accessToken = auth.accessToken;
    this.refreshToken = auth.refreshToken;
    // Use the expiresIn instead of expires, since expires is a hard to work
    // with string.
    this.refreshDeadline = new Date();
    this.refreshDeadline.setUTCSeconds(
      this.refreshDeadline.getUTCSeconds() + auth.expiresIn / 2,
    );
  }

  /**
   * Because frontegg is really slow, we explicitly wait to see our page layout, which
   * shows up once the frontegg full page loading state is complete.
   */
  async waitForFronteggToLoad() {
    await this.page.waitForSelector("[data-testid=page-layout]", {
      timeout: FRONTEGG_LOADING_TIMEOUT,
    });
  }

  /**
   * Visits a given url, signs in if necessary.
   * Sometimes frontegg just seems to hang, so we also retry on all failures.
   *
   * By default, adds ?auth_provider parameter based on the current test project's
   * auth provider configuration (set in playwright.config.ts).
   */
  async goto(
    url: string,
    options?: Parameters<Page["goto"]>[1] & { authProvider?: E2EAuthProvider },
  ) {
    const { authProvider = getProjectAuthProvider(), ...gotoOptions } =
      options || {};
    const urlWithAuth = withAuthProvider(url, authProvider);

    return retry(
      async () => {
        await this.page.goto(urlWithAuth, gotoOptions);
        const result = await Promise.race([
          (async () => {
            if (this.isOry) {
              // Unauthenticated Ory sessions redirect to the hosted
              // login on the Ory project domain.
              await this.page.waitForURL(/oryapis\.com/, {
                timeout: FRONTEGG_LOADING_TIMEOUT,
              });
            } else {
              await this.page.waitForSelector(
                "[data-test-id=input-identifier]",
                {
                  timeout: FRONTEGG_LOADING_TIMEOUT,
                },
              );
            }
            return "login";
          })(),
          (async () => {
            await this.waitForFronteggToLoad();
            return "success";
          })(),
        ]);
        if (result === "login") {
          await this.signIn();
        }
      },
      // 3 tries total, no backoff
      { retries: 2, minTimeout: 0, factor: 1 },
    );
  }

  /**
   * Make an authenticated Frontegg API request.
   */
  async fronteggRequest(path: string, request?: Partial<Options>) {
    return this.apiRequest(buildFronteggUrl(path), request);
  }

  /**
   * Make an authenticated API request.
   */
  async apiRequest(url: string, request?: Partial<Options>) {
    await this.ensureAuthenticated();
    request = {
      ...request,
      headers: {
        authorization: `Bearer ${this.accessToken}`,
        "content-type": "application/json",
        ...(request || {}).headers,
      },
    };
    return retry<Record<string, any> | null>(
      // Automatically retry network errors
      async () => {
        const response = await this.request.fetch(url, request);

        if (!response.ok()) {
          const responsePayload = await response.text();
          throw new Error(
            `API Error ${response.status()}  ${url}, req: ${
              JSON.stringify(request.data) ?? "No request body"
            }, res: ${JSON.stringify(responsePayload) ?? "No response body"}`,
          );
        }
        return response;
      },
      // No exponential backoff
      { retries: 2, minTimeout: 1000, factor: 1 },
    );
  }

  /** Block or unblock an organization **/
  async setFronteggTenantBlockedStatus(blocked: boolean) {
    if (!this.fronteggClient) {
      throw new Error("No available Frontegg client");
    }
    const { tenantId } = await this.getCurrentUser();
    await this.fronteggClient.post(
      `tenants/resources/tenants/v1/${tenantId}/metadata`,
      {
        metadata: { blocked },
      },
    );
  }

  /** Disable any existing regions. */
  async disableAllRegions() {
    await Promise.all(REGIONS.map((region) => this.disableRegion(region)));
  }

  async disableRegion(region: Region): Promise<unknown> {
    console.log(`Disabling ${region.id}, this may take up to 5min...`);
    return retry(
      async (attempts) => {
        try {
          await this.ensureAuthenticated();
          await this.request.fetch(
            `${region.url}/api/region`,
            // The timeout on the ALB is 60 seconds, so this timeout doesn't matter much
            {
              method: "DELETE",
              params: { hardDelete: "true" },
              timeout: 5 * 60000,
              headers: {
                authorization: `Bearer ${this.accessToken}`,
                "content-type": "application/json",
              },
            },
          );
        } catch (e: unknown) {
          console.error(e);
          if (e instanceof Error) {
            if (e.message.includes("API Error 504")) {
              // If we get a 504, the ALB most likely timed out
              console.log(
                `Retrying disable region for ${region.id}, attempt ${attempts}`,
              );
              return;
            }
            if (
              e.message.includes("unexpected number of bytes") ||
              e.message.includes("socket hang up") ||
              e.message.includes("ECONNRESET")
            ) {
              console.log(
                `Retrying disable region for ${region.id} after transient network error, attempt ${attempts}`,
              );
              throw e;
            }
            // If we get any other error, we should not retry
            throw new AbortError(e.message);
          }
          throw new AbortError("Unknown error occurred");
        }
      },
      // Because the ALB connection timeout is 60 seconds, we want to retry right away
      { retries: 4, minTimeout: 0, factor: 1 },
    );
  }

  async getCurrentUser(): Promise<{ id: string; tenantId: string }> {
    if (this.isOry) {
      // The token hook stamps the organization into every issued token
      // (top-level in ID tokens, under `ext` in access tokens).
      await this.ensureAuthenticated();
      const claims = jwtPayload(this.accessToken);
      const tenantId = claims.organization_id ?? claims.ext?.organization_id;
      assert(tenantId, "Ory token carries no organization_id claim");
      return { id: claims.sub, tenantId };
    }
    const response = await this.fronteggRequest(
      `/identity/resources/users/v2/me`,
    );
    assert(response);
    const { id, tenantId } = await response.json();
    return { id, tenantId };
  }

  async listAllKeys() {
    if (this.isOry) {
      const response = await this.apiRequest(
        `${CLOUD_GLOBAL_API_URL}/api/app-passwords`,
      );
      assert(response);
      return response.json();
    }
    const { id, tenantId } = await this.getCurrentUser();
    const response = await this.fronteggRequest(
      `/identity/resources/users/api-tokens/v1`,
      {
        headers: {
          "frontegg-tenant-id": tenantId,
          "frontegg-user-id": id,
        },
      },
    );
    assert(response);
    return response.json();
  }

  async deleteAllKeysOlderThan(hours: number) {
    if (this.isOry) {
      const keys = await this.listAllKeys();
      for (const k of keys) {
        const created = k.create_time ? Date.parse(k.create_time) : NaN;
        const age = new Date().getTime() - created;
        if (!Number.isFinite(age) || age < hours * 60 * 60 * 1000) {
          continue;
        }
        try {
          await this.apiRequest(
            `${CLOUD_GLOBAL_API_URL}/api/app-passwords/${k.key_id}`,
            { method: "DELETE" },
          );
        } catch (e: unknown) {
          const keyDoesNotExist =
            e instanceof Error && e.message.includes("API Error 404");
          if (!keyDoesNotExist) {
            throw e;
          }
        }
      }
      return;
    }
    const { id, tenantId } = await this.getCurrentUser();
    const userKeys = await this.listAllKeys();
    for (const k of userKeys) {
      const age = new Date().getTime() - Date.parse(k.createdAt);
      if (age < hours * 60 * 60 * 1000) {
        continue;
      }
      try {
        await this.fronteggRequest(
          `/identity/resources/users/api-tokens/v1/${k.clientId}`,
          {
            method: "DELETE",
            headers: {
              "frontegg-tenant-id": tenantId,
              "frontegg-user-id": id,
            },
          },
        );
      } catch (e: unknown) {
        // if the deployment does not exist, it's okay to ignore the error.
        const keyDoesNotExist =
          e instanceof Error && e.message.includes("API Error 404");
        if (!keyDoesNotExist) {
          throw e;
        }
      }
    }
  }
}
