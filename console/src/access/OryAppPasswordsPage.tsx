/**
 * App-passwords page for Ory-authenticated users.
 *
 * Mirrors the Frontegg AppPasswordsPage look and feel, backed by
 * Talos-issued API keys via the sync-server's /api/app-passwords
 * (see ~/queries/oryAppPasswords). Talos keys are always personal
 * (actor-scoped) — there is no service-account/roles variant yet.
 */

import { DeleteIcon } from "@chakra-ui/icons";
import {
  Alert as ChakraAlert,
  AlertDescription,
  Button,
  CloseButton,
  FormControl,
  FormErrorMessage,
  FormHelperText,
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
  useTheme,
  VStack,
} from "@chakra-ui/react";
import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { useLocation } from "react-router-dom";

import Alert from "~/components/Alert";
import { AppErrorBoundary } from "~/components/AppErrorBoundary";
import { SecretCopyableBox } from "~/components/copyableComponents";
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
  type OryAppPassword,
  useCreateOryAppPassword,
  useDeleteOryAppPassword,
  useListOryAppPasswords,
} from "~/queries/oryAppPasswords";
import { MaterializeTheme } from "~/theme";
import {
  formatDate,
  FRIENDLY_DATETIME_FORMAT_NO_SECONDS,
} from "~/utils/dateFormat";

const OryAppPasswordsPage = ({ user }: { user: User }) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const location = useLocation();

  React.useEffect(() => {
    if (location.state && "new" in location.state && location.state.new) {
      onOpen();
    }
  }, [location.pathname, location.state, onOpen]);

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
            user={user}
          />
        </AppErrorBoundary>
      </React.Suspense>
    </MainContentContainer>
  );
};

const OryAppPasswordsInner = (props: {
  isNewModalOpen: boolean;
  closeNewModal: () => void;
  user: User;
}) => {
  const { user } = props;

  const {
    mutate: createAppPassword,
    isPending: createInProgress,
    data: newPassword,
    error,
  } = useCreateOryAppPassword();

  const { data: appPasswords } = useListOryAppPasswords();

  const { register, handleSubmit, formState, reset } = useForm<{
    name: string;
  }>({
    mode: "onChange",
    defaultValues: { name: "" },
  });

  const [newPasswordClosed, setNewPasswordClosed] = useState("");

  const isSecretBoxOpen =
    newPassword &&
    newPasswordClosed !== newPassword.key_id &&
    appPasswords.map((p) => p.key_id).includes(newPassword.key_id);

  return (
    <VStack alignItems="stretch">
      {error && <Alert variant="error" message={error.message} mb="10" />}
      {isSecretBoxOpen && (
        <SecretBox
          name={newPassword.name ?? ""}
          password={newPassword.secret}
          onClose={() => setNewPasswordClosed(newPassword.key_id)}
        />
      )}
      <Text fontSize="sm" mb="2">
        App passwords allow applications and services to connect to Materialize.
      </Text>
      <OryAppPasswordsTable tokens={appPasswords} user={user} />
      <Modal
        isOpen={props.isNewModalOpen}
        onClose={props.closeNewModal}
        size="lg"
      >
        <ModalOverlay />
        <ModalContent>
          <form
            onSubmit={handleSubmit((data) => {
              createAppPassword(data.name);
              reset();
              props.closeNewModal();
            })}
          >
            <ModalHeader>New app password</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <VStack pb={6} spacing="4">
                <FormControl isInvalid={!!formState.errors.name}>
                  <FormLabel htmlFor="name" fontSize="sm">
                    Name
                  </FormLabel>
                  <Input
                    {...register("name", {
                      required: "Name is required",
                    })}
                    aria-label="Name"
                    placeholder="e.g. Personal laptop"
                    autoFocus={props.isNewModalOpen}
                    autoCorrect="off"
                    autoComplete="off"
                    size="sm"
                  />
                  <FormErrorMessage>
                    {formState.errors.name?.message}
                  </FormErrorMessage>
                  <FormHelperText>
                    Describe what you&apos;ll use the app password for, in case
                    you need to revoke it in the future. App passwords are
                    associated with your user account ({user.email}).
                  </FormHelperText>
                </FormControl>
              </VStack>
            </ModalBody>

            <ModalFooter>
              <HStack spacing="2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={props.closeNewModal}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="primary"
                  size="sm"
                  isDisabled={!!createInProgress}
                >
                  Create Password
                </Button>
              </HStack>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </VStack>
  );
};

const OryAppPasswordsTable = ({
  tokens,
  user,
}: {
  tokens: OryAppPassword[];
  user: User;
}) => {
  const { colors } = useTheme<MaterializeTheme>();

  return (
    <Table variant="standalone">
      <Thead>
        <Tr>
          <Th>Name</Th>
          <Th>User</Th>
          <Th>Created at</Th>
          <Th>Last used</Th>
          <Th />
        </Tr>
      </Thead>
      <Tbody>
        {tokens.map((token) => (
          <Tr
            key={token.key_id}
            textColor="default"
            aria-label={token.name ?? token.key_id}
          >
            <Td
              borderBottomWidth="1px"
              borderBottomColor={colors.border.primary}
            >
              {token.name}
            </Td>
            <Td
              borderBottomWidth="1px"
              borderBottomColor={colors.border.primary}
            >
              <Text color={colors.gray["500"]}>{user.email}</Text>
            </Td>
            <Td
              borderBottomWidth="1px"
              borderBottomColor={colors.border.primary}
            >
              {token.create_time
                ? formatDate(
                    new Date(token.create_time),
                    FRIENDLY_DATETIME_FORMAT_NO_SECONDS,
                  )
                : "—"}
            </Td>
            <Td
              borderBottomWidth="1px"
              borderBottomColor={colors.border.primary}
            >
              {token.last_used_time
                ? formatDate(
                    new Date(token.last_used_time),
                    FRIENDLY_DATETIME_FORMAT_NO_SECONDS,
                  )
                : "Never"}
            </Td>
            <Td
              borderBottomWidth="1px"
              borderBottomColor={colors.border.primary}
            >
              <DeleteOryAppPasswordModal token={token} />
            </Td>
          </Tr>
        ))}
        {tokens.length === 0 && (
          <Tr>
            <Td colSpan={5}>No app passwords yet.</Td>
          </Tr>
        )}
      </Tbody>
    </Table>
  );
};

const DeleteOryAppPasswordModal = ({ token }: { token: OryAppPassword }) => {
  const { mutateAsync: deleteAppPassword } = useDeleteOryAppPassword();
  const { colors } = useTheme<MaterializeTheme>();

  return (
    <DangerActionModal
      title="Delete app password"
      aria-label="Delete app password"
      colorScheme="red"
      confirmIcon={<DeleteIcon />}
      actionText=""
      finalActionText="Delete"
      confirmText={token.name ?? token.key_id}
      onConfirm={async () => {
        await deleteAppPassword({ keyId: token.key_id });
      }}
      size="sm"
      variant="outline"
    >
      <Text fontSize="sm" color={colors.foreground.primary}>
        Deleting this app password will revoke access to any devices or services
        using it to connect to Materialize.
      </Text>
    </DangerActionModal>
  );
};

const SecretBox = ({
  name,
  password,
  onClose,
}: {
  name: string;
  password: string;
  onClose: () => void;
}) => {
  const { colors } = useTheme<MaterializeTheme>();
  const obfuscatedContent = new Array(password.length).fill("*").join("");
  return (
    <ChakraAlert
      status="info"
      mb={2}
      size="sm"
      background={colors.background.info}
      borderRadius="md"
      borderWidth="1px"
      borderColor={colors.border.info}
    >
      <VStack alignItems="flex-start" width="100%">
        <AlertDescription width="100%" px={2}>
          <VStack alignItems="start">
            <Text fontSize="md" fontWeight="500">
              New password {`"${name}"`}:
            </Text>
            <SecretCopyableBox
              label="secret"
              contents={password}
              obfuscatedContent={obfuscatedContent}
            />
          </VStack>
          <Text pt={1} textStyle="text-base" color={colors.foreground.primary}>
            Write this down; you will not be able to see your app password again
            after you reload!
          </Text>
        </AlertDescription>
      </VStack>
      <CloseButton
        position="absolute"
        right={1}
        top={1}
        size="sm"
        onClick={onClose}
      />
    </ChakraAlert>
  );
};

export default OryAppPasswordsPage;
