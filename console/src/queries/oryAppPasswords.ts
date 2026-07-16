/**
 * App-password queries for Ory-authenticated users.
 *
 * Frontegg app passwords are managed against the Frontegg REST API
 * (~/queries/frontegg); under Ory they are Talos-backed API keys managed by
 * the sync-server, which proxies `/api/app-passwords` (JWT-auth,
 * actor-scoped) to the Talos admin API. These endpoints are not in the
 * generated global-api OpenAPI schema yet, so this module uses plain
 * authenticated fetches.
 */

import {
  DefaultError,
  useMutation,
  UseMutationOptions,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query";

import { apiClient } from "~/api/apiClient";
import {
  buildGlobalQueryKey,
  buildQueryKeyPart,
} from "~/api/buildQueryKeySchema";

/** An issued Talos API key, as returned by the sync-server. */
export type OryAppPassword = {
  key_id: string;
  actor_id: string;
  name?: string | null;
  status?: string | null;
  create_time?: string | null;
  expire_time?: string | null;
  last_used_time?: string | null;
};

/** Creation response: the key plus its one-time secret. */
export type NewOryAppPassword = OryAppPassword & { secret: string };

export const oryAppPasswordQueryKeys = {
  all: () => buildGlobalQueryKey("ory-app-passwords"),
  list: () =>
    [...oryAppPasswordQueryKeys.all(), buildQueryKeyPart("list")] as const,
  create: () =>
    [...oryAppPasswordQueryKeys.all(), buildQueryKeyPart("create")] as const,
  delete: () =>
    [...oryAppPasswordQueryKeys.all(), buildQueryKeyPart("delete")] as const,
};

function getCloudClient() {
  if (apiClient.type !== "cloud") {
    throw new Error("App passwords are only available in cloud mode");
  }
  return apiClient;
}

async function appPasswordsFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const client = getCloudClient();
  const response = await client.cloudApiFetch(
    `${client.cloudGlobalApiBasePath}${path}`,
    {
      headers: { "content-type": "application/json" },
      ...init,
    },
  );
  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      body && body.length < 500
        ? body
        : `Request failed with status ${response.status}`,
    );
  }
  return response;
}

export function useListOryAppPasswords() {
  return useSuspenseQuery({
    queryKey: oryAppPasswordQueryKeys.list(),
    queryFn: async ({ signal }): Promise<OryAppPassword[]> => {
      const response = await appPasswordsFetch("/api/app-passwords", {
        signal,
      });
      const keys: OryAppPassword[] = await response.json();
      return keys.toSorted(
        (x, y) =>
          Date.parse(y.create_time ?? "") - Date.parse(x.create_time ?? ""),
      );
    },
  });
}

export function useCreateOryAppPassword(
  options?: UseMutationOptions<NewOryAppPassword, DefaultError, string>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: oryAppPasswordQueryKeys.create(),
    mutationFn: async (name: string): Promise<NewOryAppPassword> => {
      const response = await appPasswordsFetch("/api/app-passwords", {
        method: "POST",
        body: JSON.stringify({ name }),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: oryAppPasswordQueryKeys.list(),
      });
    },
    ...options,
  });
}

export function useDeleteOryAppPassword(
  options: UseMutationOptions<Response, DefaultError, { keyId: string }> = {},
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: oryAppPasswordQueryKeys.delete(),
    mutationFn: async ({ keyId }: { keyId: string }) =>
      appPasswordsFetch(`/api/app-passwords/${encodeURIComponent(keyId)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: oryAppPasswordQueryKeys.list(),
      });
    },
    ...options,
  });
}
