// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Organization-member queries for Ory-backed organizations.
 *
 * Frontegg-backed organizations manage members through the embedded
 * Frontegg AdminPortal; Ory-backed organizations manage them through the
 * cloud global API. See cloud/doc/design/20260710_ory_auth_migration.md.
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
  buildInviteLink,
  inviteOrganizationMember,
  listOrganizationMembers,
  OrganizationInvite,
  OrganizationMember,
  OrganizationMemberRole,
  removeOrganizationMember,
  setOrganizationMemberRole,
} from "~/api/orgMembers";

export type { OrganizationInvite, OrganizationMember, OrganizationMemberRole };

export const memberQueryKeys = {
  all: () => buildGlobalQueryKey("organization-members"),
  list: () => [...memberQueryKeys.all(), buildQueryKeyPart("list")] as const,
  invite: () =>
    [...memberQueryKeys.all(), buildQueryKeyPart("invite")] as const,
  remove: () =>
    [...memberQueryKeys.all(), buildQueryKeyPart("remove")] as const,
  setRole: () =>
    [...memberQueryKeys.all(), buildQueryKeyPart("setRole")] as const,
};

export function useMembers() {
  return useSuspenseQuery({
    queryKey: memberQueryKeys.list(),
    queryFn: async ({ signal }) => {
      const { data: members } = await listOrganizationMembers({ signal });
      // Oldest first, so the founding admin stays at the top.
      return members.toSorted(
        (x, y) => Date.parse(x.joinedAt) - Date.parse(y.joinedAt),
      );
    },
  });
}

export type NewInvite = OrganizationInvite & {
  /** The email the invite was issued for. */
  email: string;
  /** Shareable acceptance link; no email is sent by the server. */
  inviteLink: string;
};

export function useInviteMember(
  options?: UseMutationOptions<
    NewInvite,
    DefaultError,
    { email: string; role: OrganizationMemberRole }
  >,
) {
  return useMutation({
    mutationKey: memberQueryKeys.invite(),
    mutationFn: async ({
      email,
      role,
    }: {
      email: string;
      role: OrganizationMemberRole;
    }) => {
      const { data: invite } = await inviteOrganizationMember({ email, role });
      return {
        ...invite,
        email,
        inviteLink: buildInviteLink(invite.inviteToken),
      };
    },
    ...options,
  });
}

export function useRemoveMember(
  options?: UseMutationOptions<void, DefaultError, { memberId: string }>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: memberQueryKeys.remove(),
    mutationFn: async ({ memberId }: { memberId: string }) => {
      await removeOrganizationMember(memberId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: memberQueryKeys.list(),
      });
    },
    ...options,
  });
}

export function useSetMemberRole(
  options?: UseMutationOptions<
    void,
    DefaultError,
    { memberId: string; role: OrganizationMemberRole }
  >,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: memberQueryKeys.setRole(),
    mutationFn: async ({
      memberId,
      role,
    }: {
      memberId: string;
      role: OrganizationMemberRole;
    }) => {
      await setOrganizationMemberRole(memberId, role);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: memberQueryKeys.list(),
      });
    },
    ...options,
  });
}
