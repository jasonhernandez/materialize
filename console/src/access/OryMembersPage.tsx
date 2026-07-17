// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * Organization-member management for Ory-backed organizations.
 *
 * Frontegg-backed organizations manage members through the embedded
 * Frontegg AdminPortal (Account settings → Users); Ory-backed
 * organizations use this console-native page, backed by the cloud global
 * API. The switch happens in MembersRoutes based on the session's token
 * issuer.
 *
 * Admins (MaterializePlatformAdmin role) can invite members, change
 * roles, and remove members; regular members see a read-only list.
 */

import { DeleteIcon } from "@chakra-ui/icons";
import {
  Button,
  FormControl,
  FormErrorMessage,
  FormLabel,
  HStack,
  Input,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Select,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useDisclosure,
  VStack,
} from "@chakra-ui/react";
import React, { useState } from "react";
import { useForm } from "react-hook-form";

import { isSuperUser } from "~/api/auth";
import { OpenApiFetchError } from "~/api/OpenApiFetchError";
import Alert from "~/components/Alert";
import { AppErrorBoundary } from "~/components/AppErrorBoundary";
import { CopyableBox } from "~/components/copyableComponents";
import DangerActionModal from "~/components/DangerActionModal";
import { LoadingContainer } from "~/components/LoadingContainer";
import { Modal } from "~/components/Modal";
import { User } from "~/external-library-wrappers/frontegg";
import {
  MainContentContainer,
  PageHeader,
  PageHeading,
} from "~/layouts/BaseLayout";
import {
  OrganizationMember,
  OrganizationMemberRole,
  useInviteMember,
  useMembers,
  useRemoveMember,
  useSetMemberRole,
} from "~/queries/oryMembers";
import {
  formatDate,
  FRIENDLY_DATETIME_FORMAT_NO_SECONDS,
} from "~/utils/dateFormat";

const formatTime = (time: string | null | undefined) =>
  time ? formatDate(new Date(time), FRIENDLY_DATETIME_FORMAT_NO_SECONDS) : "-";

const ROLE_LABELS: Record<OrganizationMemberRole, string> = {
  admin: "Admin",
  member: "Member",
};

const inviteErrorMessage = (error: Error) => {
  if (error instanceof OpenApiFetchError && error.status === 409) {
    return "That email is already a member of this organization.";
  }
  return error.message;
};

const MemberRow = ({
  member,
  isAdmin,
  isSelf,
}: {
  member: OrganizationMember;
  isAdmin: boolean;
  isSelf: boolean;
}) => {
  const { mutateAsync: removeMember } = useRemoveMember();
  const { mutate: setMemberRole, isPending: setRoleInProgress } =
    useSetMemberRole();

  return (
    <Tr aria-label={member.email}>
      <Td>{member.email}</Td>
      <Td>
        {/* Admins can change other members' roles; nobody edits their own
            row (the backend rejects self-demotion/removal anyway). */}
        {isAdmin && !isSelf ? (
          <Select
            aria-label={`Role for ${member.email}`}
            size="sm"
            maxW="40"
            value={member.role}
            isDisabled={setRoleInProgress}
            onChange={(e) =>
              setMemberRole({
                memberId: member.id,
                role: e.target.value as OrganizationMemberRole,
              })
            }
          >
            <option value="admin">{ROLE_LABELS.admin}</option>
            <option value="member">{ROLE_LABELS.member}</option>
          </Select>
        ) : (
          ROLE_LABELS[member.role]
        )}
      </Td>
      <Td>{formatTime(member.joinedAt)}</Td>
      {isAdmin && (
        <Td textAlign="right">
          {!isSelf && (
            <DangerActionModal
              title="Remove member"
              aria-label="Remove member"
              colorScheme="red"
              confirmIcon={<DeleteIcon />}
              actionText=""
              finalActionText="Remove"
              confirmText={member.email}
              onConfirm={async () => {
                await removeMember({ memberId: member.id });
              }}
              size="sm"
              variant="outline"
            >
              <Text fontSize="sm">
                Removing this member will revoke their access to this
                organization and everything in it.
              </Text>
            </DangerActionModal>
          )}
        </Td>
      )}
    </Tr>
  );
};

const OryMembersInner = ({
  user,
  isAdmin,
  isInviteModalOpen,
  closeInviteModal,
}: {
  user: User;
  isAdmin: boolean;
  isInviteModalOpen: boolean;
  closeInviteModal: () => void;
}) => {
  const { data: members } = useMembers();
  const {
    mutate: inviteMember,
    isPending: inviteInProgress,
    data: newInvite,
    error,
  } = useInviteMember();
  const [inviteBoxClosed, setInviteBoxClosed] = useState("");

  const { register, handleSubmit, formState, reset } = useForm<{
    email: string;
    role: OrganizationMemberRole;
  }>({
    mode: "onChange",
    defaultValues: { email: "", role: "member" },
  });

  const isInviteBoxOpen =
    newInvite && inviteBoxClosed !== newInvite.inviteToken;

  return (
    <VStack alignItems="stretch">
      {error && (
        <Alert variant="error" message={inviteErrorMessage(error)} mb="10" />
      )}
      {isInviteBoxOpen && (
        <VStack alignItems="stretch" mb="2" spacing="2">
          <Text fontSize="md" fontWeight="500">
            Invite link for {`"${newInvite.email}"`}
          </Text>
          <CopyableBox
            aria-label="Invite link"
            contents={newInvite.inviteLink}
          />
          <HStack justifyContent="space-between">
            <Text fontSize="sm">
              No email is sent — share this link with the invitee yourself. It
              expires {formatTime(newInvite.expiresAt)}.
            </Text>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setInviteBoxClosed(newInvite.inviteToken)}
            >
              Dismiss
            </Button>
          </HStack>
        </VStack>
      )}
      <Text fontSize="sm" mb="2">
        Members can sign in to this organization&apos;s Materialize console.
      </Text>
      <Table variant="simple">
        <Thead>
          <Tr>
            <Th>Email</Th>
            <Th>Role</Th>
            <Th>Joined</Th>
            {isAdmin && <Th />}
          </Tr>
        </Thead>
        <Tbody>
          {members.map((member) => (
            <MemberRow
              key={member.id}
              member={member}
              isAdmin={isAdmin}
              isSelf={member.id === user.id}
            />
          ))}
        </Tbody>
      </Table>
      <Modal isOpen={isInviteModalOpen} onClose={closeInviteModal} size="lg">
        <ModalOverlay />
        <ModalContent>
          <form
            onSubmit={handleSubmit((data) => {
              inviteMember({ email: data.email, role: data.role });
              reset();
              closeInviteModal();
            })}
          >
            <ModalHeader>Invite member</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <VStack spacing="4">
                <FormControl isInvalid={!!formState.errors.email}>
                  <FormLabel htmlFor="email" fontSize="sm">
                    Email
                  </FormLabel>
                  <Input
                    {...register("email", {
                      required: "Email is required.",
                      pattern: {
                        value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                        message: "Must be a valid email address.",
                      },
                    })}
                    id="email"
                    autoFocus
                    placeholder="e.g. teammate@example.com"
                    size="sm"
                  />
                  <FormErrorMessage>
                    {formState.errors.email?.message}
                  </FormErrorMessage>
                </FormControl>
                <FormControl>
                  <FormLabel htmlFor="role" fontSize="sm">
                    Role
                  </FormLabel>
                  <Select {...register("role")} id="role" size="sm">
                    <option value="member">{ROLE_LABELS.member}</option>
                    <option value="admin">{ROLE_LABELS.admin}</option>
                  </Select>
                </FormControl>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <HStack>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={closeInviteModal}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  type="submit"
                  isDisabled={!formState.isValid}
                  isLoading={inviteInProgress}
                >
                  Create invite
                </Button>
              </HStack>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </VStack>
  );
};

const OryMembersPage = ({ user }: { user: User }) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  // Only UI gating; the backend enforces the real permissions on every
  // members/invites call from the identity's metadata.
  const isAdmin = isSuperUser(user);

  return (
    <MainContentContainer>
      <PageHeader>
        <PageHeading>Members</PageHeading>
        {isAdmin && (
          <Button variant="primary" size="sm" onClick={onOpen}>
            Invite member
          </Button>
        )}
      </PageHeader>
      <React.Suspense fallback={<LoadingContainer />}>
        <AppErrorBoundary>
          <OryMembersInner
            user={user}
            isAdmin={isAdmin}
            isInviteModalOpen={isOpen}
            closeInviteModal={onClose}
          />
        </AppErrorBoundary>
      </React.Suspense>
    </MainContentContainer>
  );
};

export default OryMembersPage;
