// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * App-password queries for Ory-backed organizations.
 *
 * Frontegg-backed organizations manage app passwords through the Frontegg
 * api-tokens API (see ~/queries/frontegg); Ory-backed organizations manage
 * them through the cloud global API, which proxies to Ory Talos. See
 * cloud/doc/design/20260710_ory_auth_migration.md.
 */

import {
  DefaultError,
  useMutation,
  UseMutationOptions,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query";

import {
  buildGlobalQueryKey,
  buildQueryKeyPart,
} from "~/api/buildQueryKeySchema";
import {
  AppPassword,
  createAppPassword,
  CreateAppPasswordResponse,
  deleteAppPassword,
  listAppPasswords,
} from "~/api/cloudGlobalApi";
import { obfuscateSecret } from "~/utils/format";

export type { AppPassword };

export const appPasswordQueryKeys = {
  all: () => buildGlobalQueryKey("app-passwords"),
  list: () =>
    [...appPasswordQueryKeys.all(), buildQueryKeyPart("list")] as const,
  create: () =>
    [...appPasswordQueryKeys.all(), buildQueryKeyPart("create")] as const,
  delete: () =>
    [...appPasswordQueryKeys.all(), buildQueryKeyPart("delete")] as const,
};

export function useListAppPasswords() {
  return useSuspenseQuery({
    queryKey: appPasswordQueryKeys.list(),
    queryFn: async ({ signal }) => {
      const { data: appPasswords } = await listAppPasswords({ signal });
      return appPasswords.toSorted(
        (x, y) =>
          Date.parse(y.create_time ?? "") - Date.parse(x.create_time ?? ""),
      );
    },
  });
}

export type NewAppPassword = CreateAppPasswordResponse & {
  obfuscatedPassword: string;
};

export function useCreateAppPassword(
  options?: UseMutationOptions<NewAppPassword, DefaultError, { name: string }>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: appPasswordQueryKeys.create(),
    mutationFn: async ({ name }: { name: string }) => {
      const { data: created } = await createAppPassword(name);
      return {
        ...created,
        obfuscatedPassword: obfuscateSecret(created.secret),
      };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: appPasswordQueryKeys.list(),
      });
    },
    ...options,
  });
}

export function useDeleteAppPassword(
  options?: UseMutationOptions<void, DefaultError, { keyId: string }>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: appPasswordQueryKeys.delete(),
    mutationFn: async ({ keyId }: { keyId: string }) => {
      await deleteAppPassword(keyId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: appPasswordQueryKeys.list(),
      });
    },
    ...options,
  });
}
