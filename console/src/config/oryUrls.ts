/**
 * Ory Network URL configuration helpers.
 *
 * These functions determine the Ory project URLs based on the current stack/environment.
 * During the migration period, Ory runs alongside Frontegg with feature flags controlling routing.
 */

/**
 * Get the Ory project URL for a given stack.
 *
 * URL patterns:
 * - Environment variable override: VITE_ORY_PROJECT_URL (highest priority;
 *   required for personal stacks)
 * - Production: https://auth.cloud.materialize.com (custom domain)
 * - Staging: https://auth.staging.cloud.materialize.com (custom domain)
 * - Local: Uses staging Ory project
 *
 * Personal stacks have no derivable URL: Ory Network assigns random project
 * slugs (e.g. funny-edison-9unu847vwo.projects.oryapis.com), so
 * VITE_ORY_PROJECT_URL must be set in .env.local (copy the
 * `ory_project_url` stack output; see doc/playbooks/ory-personal-stack.md
 * in the cloud repo).
 */
export function getOryProjectUrl(stack: string): string {
  // Environment variable override (mandatory for personal stacks).
  const envOverride =
    typeof import.meta.env !== "undefined"
      ? import.meta.env.VITE_ORY_PROJECT_URL
      : undefined;

  if (envOverride) {
    return envOverride;
  }

  switch (stack) {
    case "production":
      // Production uses custom domain once Ory is fully deployed
      // During migration, this may still point to Ory Network domain
      return "https://auth.cloud.materialize.com";
    case "staging":
      return "https://auth.staging.cloud.materialize.com";
    case "local":
      // Local development uses staging Ory project
      return "https://auth.staging.cloud.materialize.com";
    default:
      // This is called eagerly at AppConfig construction, so don't throw —
      // Frontegg-backed stacks never use the value. Ory flows against this
      // empty URL fail immediately and this warning explains why.
      console.warn(
        `No Ory project URL for personal stack "${stack}": Ory Network ` +
          "slugs are random and cannot be derived. Set VITE_ORY_PROJECT_URL " +
          "in .env.local to this stack's ory_project_url Pulumi output.",
      );
      return "";
  }
}

/**
 * Get the Ory OAuth2 client ID for a given stack.
 *
 * Client IDs are configured per-environment and stored in environment variables.
 * Returns empty string if not configured (Ory not yet set up for this stack).
 */
export function getOryOAuth2ClientId(stack: string): string {
  // Check for stack-specific environment variable first
  const stackKey = stack.toUpperCase().replace(/-/g, "_");
  const stackSpecificId =
    typeof import.meta.env !== "undefined"
      ? import.meta.env[`VITE_ORY_OAUTH2_CLIENT_ID_${stackKey}`]
      : undefined;

  if (stackSpecificId) {
    return stackSpecificId;
  }

  // Fall back to generic environment variable
  const genericId =
    typeof import.meta.env !== "undefined"
      ? import.meta.env.VITE_ORY_OAUTH2_CLIENT_ID
      : undefined;

  return genericId ?? "";
}

/**
 * Get the JWK (JSON Web Key) URL for validating Ory tokens.
 */
export function getOryJwkUrl(stack: string): string {
  const projectUrl = getOryProjectUrl(stack);
  return `${projectUrl}/.well-known/jwks.json`;
}
