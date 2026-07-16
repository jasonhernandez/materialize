// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

/**
 * App-password management for Ory-backed organizations, where app
 * passwords are Ory Talos API keys managed through the cloud global API.
 * Frontegg-backed organizations use AppPasswordsPage instead; the switch
 * happens in AppPasswordRoutes based on the session's token issuer.
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

import Alert from "~/components/Alert";
import { AppErrorBoundary } from "~/components/AppErrorBoundary";
import { SecretCopyableBox } from "~/components/copyableComponents";
import DangerActionModal from "~/components/DangerActionModal";
import { LoadingContainer } from "~/components/LoadingContainer";
import { Modal } from "~/components/Modal";
import {
  MainContentContainer,
  PageHeader,
  PageHeading,
} from "~/layouts/BaseLayout";
import {
  AppPassword,
  useCreateAppPassword,
  useDeleteAppPassword,
  useListAppPasswords,
} from "~/queries/appPasswords";
import {
  formatDate,
  FRIENDLY_DATETIME_FORMAT_NO_SECONDS,
} from "~/utils/dateFormat";

const formatTime = (time: string | null | undefined) =>
  time ? formatDate(new Date(time), FRIENDLY_DATETIME_FORMAT_NO_SECONDS) : "-";

const AppPasswordRow = ({ appPassword }: { appPassword: AppPassword }) => {
  const { mutateAsync: deleteAppPassword } = useDeleteAppPassword();
  const name = appPassword.name ?? appPassword.key_id;
  return (
    <Tr aria-label={name}>
      <Td>{name}</Td>
      <Td>{formatTime(appPassword.create_time)}</Td>
      <Td>{formatTime(appPassword.last_used_time)}</Td>
      <Td textAlign="right">
        <DangerActionModal
          title="Delete app password"
          aria-label="Delete app password"
          colorScheme="red"
          confirmIcon={<DeleteIcon />}
          actionText=""
          finalActionText="Delete"
          confirmText={name}
          onConfirm={async () => {
            await deleteAppPassword({ keyId: appPassword.key_id });
          }}
          size="sm"
          variant="outline"
        >
          <Text fontSize="sm">
            Deleting this app password will revoke access to any devices or
            services using it to connect to Materialize.
          </Text>
        </DangerActionModal>
      </Td>
    </Tr>
  );
};

const OryAppPasswordsInner = ({
  isNewModalOpen,
  closeNewModal,
}: {
  isNewModalOpen: boolean;
  closeNewModal: () => void;
}) => {
  const { data: appPasswords } = useListAppPasswords();
  const {
    mutate: createAppPassword,
    isPending: createInProgress,
    data: newPassword,
    error,
  } = useCreateAppPassword();
  const [secretBoxClosed, setSecretBoxClosed] = useState("");

  const { register, handleSubmit, formState, reset } = useForm<{
    name: string;
  }>({
    mode: "onChange",
    defaultValues: { name: "" },
  });

  const isSecretBoxOpen = newPassword && secretBoxClosed !== newPassword.key_id;

  return (
    <VStack alignItems="stretch">
      {error && <Alert variant="error" message={error.message} mb="10" />}
      {isSecretBoxOpen && (
        <VStack alignItems="stretch" mb="2" spacing="2">
          <Text fontSize="md" fontWeight="500">
            New app password {`"${newPassword.name ?? ""}"`}
          </Text>
          <SecretCopyableBox
            label="app password"
            contents={newPassword.secret}
            obfuscatedContent={newPassword.obfuscatedPassword}
          />
          <HStack justifyContent="space-between">
            <Text fontSize="sm">
              Write this down; you will not be able to see your credentials
              again after you reload!
            </Text>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setSecretBoxClosed(newPassword.key_id)}
            >
              Dismiss
            </Button>
          </HStack>
        </VStack>
      )}
      <Text fontSize="sm" mb="2">
        App passwords allow applications and services to connect to Materialize.
      </Text>
      <Table variant="simple">
        <Thead>
          <Tr>
            <Th>Name</Th>
            <Th>Created</Th>
            <Th>Last used</Th>
            <Th />
          </Tr>
        </Thead>
        <Tbody>
          {appPasswords.map((appPassword) => (
            <AppPasswordRow
              key={appPassword.key_id}
              appPassword={appPassword}
            />
          ))}
        </Tbody>
      </Table>
      <Modal isOpen={isNewModalOpen} onClose={closeNewModal} size="lg">
        <ModalOverlay />
        <ModalContent>
          <form
            onSubmit={handleSubmit((data) => {
              createAppPassword({ name: data.name });
              reset();
              closeNewModal();
            })}
          >
            <ModalHeader>New app password</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <FormControl isInvalid={!!formState.errors.name}>
                <FormLabel htmlFor="name" fontSize="sm">
                  Name
                </FormLabel>
                <Input
                  {...register("name", {
                    required: "Name is required.",
                    maxLength: {
                      value: 256,
                      message: "Name must not exceed 256 characters.",
                    },
                  })}
                  autoFocus
                  placeholder="e.g. production-dashboard"
                  size="sm"
                />
                <FormErrorMessage>
                  {formState.errors.name?.message}
                </FormErrorMessage>
              </FormControl>
            </ModalBody>
            <ModalFooter>
              <HStack>
                <Button variant="secondary" size="sm" onClick={closeNewModal}>
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  type="submit"
                  isDisabled={!formState.isValid}
                  isLoading={createInProgress}
                >
                  Create app password
                </Button>
              </HStack>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </VStack>
  );
};

const OryAppPasswordsPage = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <MainContentContainer>
      <PageHeader>
        <PageHeading>App Passwords</PageHeading>
        <Button variant="primary" size="sm" onClick={onOpen}>
          New app password
        </Button>
      </PageHeader>
      <React.Suspense fallback={<LoadingContainer />}>
        <AppErrorBoundary>
          <OryAppPasswordsInner
            isNewModalOpen={isOpen}
            closeNewModal={onClose}
          />
        </AppErrorBoundary>
      </React.Suspense>
    </MainContentContainer>
  );
};

export default OryAppPasswordsPage;
