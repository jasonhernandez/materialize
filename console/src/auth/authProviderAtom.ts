/**
 * Auth Provider State Atom
 *
 * Jotai atom that holds the detected authentication provider.
 * This is set once at app startup before authentication begins.
 */

import { atom } from "jotai";

import { type AuthProviderType } from "./detectAuthProvider";

/**
 * Atom holding the current auth provider type.
 * Defaults to "frontegg" and is set during app initialization.
 *
 * This atom is set ONCE at startup by AuthProviderWrapper and should not
 * be modified during the app lifecycle (except during logout/re-auth).
 */
export const authProviderAtom = atom<AuthProviderType>("frontegg");

/**
 * Atom indicating whether auth provider detection is complete.
 * Used to show loading state during initial detection.
 */
export const authProviderDetectedAtom = atom<boolean>(false);
