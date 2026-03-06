// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! KMS-based envelope encryption for S3 data at rest.
//!
//! Uses AWS KMS to generate Data Encryption Keys (DEKs), then encrypts data
//! locally with AES-256-GCM. Each encrypted object is self-contained: it
//! includes the KMS-wrapped DEK so only the master key ARN is needed to decrypt.
//!
//! Encrypted object byte layout:
//! ```text
//! [version: 1 byte (0x01)] || [wrapped_dek_len: 2 bytes LE] || [wrapped_dek: N bytes]
//! || [nonce: 12 bytes] || [ciphertext + GCM tag: variable]
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use aws_sdk_kms::primitives::Blob as KmsBlob;
use aws_sdk_kms::types::DataKeySpec;
use aws_sdk_kms::Client as KmsClient;
use bytes::Bytes;
use mz_ore::bytes::SegmentedBytes;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use aws_lc_rs::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM, NONCE_LEN};

use crate::location::{Blob, BlobMetadata, ExternalError};

const ENVELOPE_VERSION: u8 = 0x01;
const WRAPPED_DEK_LEN_SIZE: usize = 2;
const GCM_TAG_LEN: usize = 16;

struct DataEncryptionKey {
    plaintext: [u8; 32],
    wrapped: Vec<u8>,
}

/// Manages KMS-derived data encryption keys with background rotation.
pub struct EnvelopeEncryption {
    kms_client: KmsClient,
    kms_key_id: String,
    current_dek: Arc<RwLock<DataEncryptionKey>>,
}

impl std::fmt::Debug for EnvelopeEncryption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvelopeEncryption")
            .field("kms_key_id", &self.kms_key_id)
            .finish_non_exhaustive()
    }
}

impl EnvelopeEncryption {
    /// Initialize: call KMS GenerateDataKey, cache result.
    pub async fn new(kms_client: KmsClient, kms_key_id: String) -> Result<Self, anyhow::Error> {
        let dek = Self::generate_dek(&kms_client, &kms_key_id).await?;
        info!(kms_key_id = %kms_key_id, "envelope encryption initialized");
        Ok(EnvelopeEncryption {
            kms_client,
            kms_key_id,
            current_dek: Arc::new(RwLock::new(dek)),
        })
    }

    #[cfg(test)]
    fn new_test(key: [u8; 32], wrapped_dek: Vec<u8>) -> Self {
        use aws_sdk_kms::config::Builder;
        let kms_config = Builder::new()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .build();
        let kms_client = KmsClient::from_conf(kms_config);
        EnvelopeEncryption {
            kms_client,
            kms_key_id: String::new(),
            current_dek: Arc::new(RwLock::new(DataEncryptionKey {
                plaintext: key,
                wrapped: wrapped_dek,
            })),
        }
    }

    async fn generate_dek(
        kms_client: &KmsClient,
        kms_key_id: &str,
    ) -> Result<DataEncryptionKey, anyhow::Error> {
        let resp = kms_client
            .generate_data_key()
            .key_id(kms_key_id)
            .key_spec(DataKeySpec::Aes256)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("KMS GenerateDataKey failed: {}", e))?;

        let plaintext_blob = resp
            .plaintext()
            .ok_or_else(|| anyhow::anyhow!("KMS GenerateDataKey returned no plaintext"))?;
        let wrapped_blob = resp
            .ciphertext_blob()
            .ok_or_else(|| anyhow::anyhow!("KMS GenerateDataKey returned no ciphertext_blob"))?;

        let plaintext_bytes = plaintext_blob.as_ref();
        if plaintext_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "KMS returned key of length {}, expected 32",
                plaintext_bytes.len()
            ));
        }

        let mut key = [0u8; 32];
        key.copy_from_slice(plaintext_bytes);

        Ok(DataEncryptionKey {
            plaintext: key,
            wrapped: wrapped_blob.as_ref().to_vec(),
        })
    }

    /// Spawn background rotation task (every `interval`).
    pub fn start_rotation(self: &Arc<Self>, interval: Duration) -> JoinHandle<()> {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip first immediate tick
            loop {
                ticker.tick().await;
                match Self::generate_dek(&this.kms_client, &this.kms_key_id).await {
                    Ok(new_dek) => {
                        *this.current_dek.write().await = new_dek;
                        info!("DEK rotated successfully");
                    }
                    Err(e) => {
                        warn!("DEK rotation failed, keeping current key: {}", e);
                    }
                }
            }
        })
    }

    /// Encrypt plaintext. Reads cached DEK (RwLock read — uncontended).
    /// Returns: version || wrapped_dek_len || wrapped_dek || nonce || ciphertext || tag
    pub async fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, anyhow::Error> {
        let dek = self.current_dek.read().await;
        encrypt_with_dek(&dek.plaintext, &dek.wrapped, plaintext)
    }

    /// Decrypt ciphertext. Parses wrapped DEK from header, uses cached key if
    /// it matches, otherwise calls KMS Decrypt for the wrapped DEK.
    pub async fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>, anyhow::Error> {
        let (wrapped_dek, nonce_and_ciphertext) = parse_envelope(encrypted)?;

        // Fast path: check if wrapped DEK matches our cached key.
        let dek = self.current_dek.read().await;
        let plaintext_key = if dek.wrapped == wrapped_dek {
            dek.plaintext
        } else {
            drop(dek);
            debug!("wrapped DEK mismatch, calling KMS Decrypt for old DEK");
            self.decrypt_dek(wrapped_dek).await?
        };

        decrypt_with_key(&plaintext_key, nonce_and_ciphertext)
    }

    async fn decrypt_dek(&self, wrapped_dek: &[u8]) -> Result<[u8; 32], anyhow::Error> {
        let resp = self
            .kms_client
            .decrypt()
            .key_id(&self.kms_key_id)
            .ciphertext_blob(KmsBlob::new(wrapped_dek))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("KMS Decrypt failed: {}", e))?;

        let plaintext_blob = resp
            .plaintext()
            .ok_or_else(|| anyhow::anyhow!("KMS Decrypt returned no plaintext"))?;
        let bytes = plaintext_blob.as_ref();
        if bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "KMS Decrypt returned key of length {}, expected 32",
                bytes.len()
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Ok(key)
    }
}

/// Encrypt using a raw AES-256-GCM key. Builds the envelope format.
pub fn encrypt_with_dek(
    key: &[u8; 32],
    wrapped_dek: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, anyhow::Error> {
    let unbound = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| anyhow::anyhow!("failed to create AES-256-GCM key"))?;
    let aead_key = LessSafeKey::new(unbound);

    let mut nonce_bytes = [0u8; NONCE_LEN];
    aws_lc_rs::rand::fill(&mut nonce_bytes)
        .map_err(|_| anyhow::anyhow!("failed to generate random nonce"))?;
    let nonce = Nonce::assume_unique_for_key(nonce_bytes);

    // in_out buffer: plaintext + space for GCM tag
    let mut in_out = Vec::with_capacity(plaintext.len() + GCM_TAG_LEN);
    in_out.extend_from_slice(plaintext);

    aead_key
        .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
        .map_err(|_| anyhow::anyhow!("AES-256-GCM seal failed"))?;

    let wrapped_len = wrapped_dek.len() as u16;
    let header_size = 1 + WRAPPED_DEK_LEN_SIZE + wrapped_dek.len() + NONCE_LEN;
    let mut output = Vec::with_capacity(header_size + in_out.len());
    output.push(ENVELOPE_VERSION);
    output.extend_from_slice(&wrapped_len.to_le_bytes());
    output.extend_from_slice(wrapped_dek);
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&in_out);

    Ok(output)
}

/// Parse the envelope header, returning (wrapped_dek, nonce_and_ciphertext_with_tag).
pub fn parse_envelope(data: &[u8]) -> Result<(&[u8], &[u8]), anyhow::Error> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("encrypted data is empty"));
    }
    if data[0] != ENVELOPE_VERSION {
        return Err(anyhow::anyhow!(
            "unsupported envelope version: 0x{:02x}",
            data[0]
        ));
    }
    let min_header = 1 + WRAPPED_DEK_LEN_SIZE;
    if data.len() < min_header {
        return Err(anyhow::anyhow!("encrypted data too short for header"));
    }
    let wrapped_len = u16::from_le_bytes([data[1], data[2]]) as usize;
    let wrapped_end = min_header + wrapped_len;
    let payload_start = wrapped_end + NONCE_LEN;
    if data.len() < payload_start + GCM_TAG_LEN {
        return Err(anyhow::anyhow!("encrypted data too short for envelope"));
    }
    Ok((&data[min_header..wrapped_end], &data[wrapped_end..]))
}

/// Decrypt using a raw AES-256-GCM key. Input is nonce || ciphertext || tag.
pub fn decrypt_with_key(
    key: &[u8; 32],
    nonce_and_ciphertext: &[u8],
) -> Result<Vec<u8>, anyhow::Error> {
    if nonce_and_ciphertext.len() < NONCE_LEN + GCM_TAG_LEN {
        return Err(anyhow::anyhow!("ciphertext too short"));
    }
    let (nonce_bytes, ciphertext_with_tag) = nonce_and_ciphertext.split_at(NONCE_LEN);

    let unbound = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| anyhow::anyhow!("failed to create AES-256-GCM key"))?;
    let aead_key = LessSafeKey::new(unbound);

    let mut nonce_arr = [0u8; NONCE_LEN];
    nonce_arr.copy_from_slice(nonce_bytes);
    let nonce = Nonce::assume_unique_for_key(nonce_arr);

    let mut buf = ciphertext_with_tag.to_vec();
    let plaintext = aead_key
        .open_in_place(nonce, Aad::empty(), &mut buf)
        .map_err(|_| {
            anyhow::anyhow!("AES-256-GCM authentication failed: data may be tampered")
        })?;

    Ok(plaintext.to_vec())
}

/// Configuration for blob-level envelope encryption.
#[derive(Debug, Clone)]
pub struct BlobEncryptionConfig {
    /// The KMS key ARN to use for envelope encryption.
    pub kms_key_id: String,
    /// The AWS region for the KMS key. Falls back to the blob region if unset.
    pub kms_region: Option<String>,
    /// Optional endpoint override (e.g. for LocalStack).
    pub endpoint: Option<String>,
    /// Optional IAM role ARN to assume before calling KMS.
    pub role_arn: Option<String>,
    /// How often to rotate the data encryption key.
    pub dek_rotation_interval: Duration,
}

impl BlobEncryptionConfig {
    /// Build a KMS client from this config.
    pub async fn build_kms_client(&self) -> Result<KmsClient, ExternalError> {
        let mut loader = mz_aws_util::defaults();

        if let Some(region) = &self.kms_region {
            loader = loader.region(aws_config::Region::new(region.clone()));
        }
        if let Some(endpoint) = &self.endpoint {
            loader = loader.endpoint_url(endpoint);
        }

        let sdk_config = loader.load().await;
        Ok(KmsClient::new(&sdk_config))
    }
}

/// A [Blob] wrapper that transparently encrypts on `set()` and decrypts on `get()`.
pub struct EncryptedBlob {
    inner: Arc<dyn Blob>,
    encryption: Arc<EnvelopeEncryption>,
    _rotation_handle: JoinHandle<()>,
}

impl std::fmt::Debug for EncryptedBlob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedBlob").finish_non_exhaustive()
    }
}

impl EncryptedBlob {
    /// Create a new `EncryptedBlob` wrapping `inner` with KMS envelope encryption.
    pub async fn new(
        inner: Arc<dyn Blob>,
        kms_client: KmsClient,
        kms_key_id: String,
        rotation_interval: Duration,
    ) -> Result<Self, ExternalError> {
        let encryption = Arc::new(
            EnvelopeEncryption::new(kms_client, kms_key_id)
                .await
                .map_err(ExternalError::from)?,
        );
        let rotation_handle = encryption.start_rotation(rotation_interval);
        Ok(EncryptedBlob {
            inner,
            encryption,
            _rotation_handle: rotation_handle,
        })
    }

    #[cfg(test)]
    fn new_test(inner: Arc<dyn Blob>) -> Self {
        let encryption = Arc::new(EnvelopeEncryption::new_test(
            [0x42u8; 32],
            vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE],
        ));
        // No rotation in tests — use a no-op handle.
        let rotation_handle = tokio::spawn(async {});
        EncryptedBlob {
            inner,
            encryption,
            _rotation_handle: rotation_handle,
        }
    }
}

#[async_trait]
impl Blob for EncryptedBlob {
    async fn get(&self, key: &str) -> Result<Option<SegmentedBytes>, ExternalError> {
        let maybe_segments = self.inner.get(key).await?;
        match maybe_segments {
            None => Ok(None),
            Some(segments) => {
                let encrypted = segments.into_contiguous();
                let plaintext = self
                    .encryption
                    .decrypt(&encrypted)
                    .await
                    .map_err(ExternalError::from)?;
                Ok(Some(SegmentedBytes::from(plaintext)))
            }
        }
    }

    async fn list_keys_and_metadata(
        &self,
        key_prefix: &str,
        f: &mut (dyn FnMut(BlobMetadata) + Send + Sync),
    ) -> Result<(), ExternalError> {
        self.inner.list_keys_and_metadata(key_prefix, f).await
    }

    async fn set(&self, key: &str, value: Bytes) -> Result<(), ExternalError> {
        let ciphertext = self
            .encryption
            .encrypt(&value)
            .await
            .map_err(ExternalError::from)?;
        self.inner.set(key, Bytes::from(ciphertext)).await
    }

    async fn delete(&self, key: &str) -> Result<Option<usize>, ExternalError> {
        self.inner.delete(key).await
    }

    async fn restore(&self, key: &str) -> Result<(), ExternalError> {
        self.inner.restore(key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        [0x42u8; 32]
    }

    fn test_wrapped_dek() -> Vec<u8> {
        vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]
    }

    #[test]
    fn roundtrip() {
        let key = test_key();
        let wrapped = test_wrapped_dek();
        let plaintext = b"hello, envelope encryption!";

        let encrypted = encrypt_with_dek(&key, &wrapped, plaintext).unwrap();
        assert_ne!(&encrypted[..], plaintext);

        let (parsed_wrapped, nonce_ct) = parse_envelope(&encrypted).unwrap();
        assert_eq!(parsed_wrapped, &wrapped[..]);

        let decrypted = decrypt_with_key(&key, nonce_ct).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn roundtrip_empty_plaintext() {
        let key = test_key();
        let wrapped = test_wrapped_dek();

        let encrypted = encrypt_with_dek(&key, &wrapped, b"").unwrap();
        let (_, nonce_ct) = parse_envelope(&encrypted).unwrap();
        let decrypted = decrypt_with_key(&key, nonce_ct).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn tamper_detection() {
        let key = test_key();
        let wrapped = test_wrapped_dek();

        let mut encrypted = encrypt_with_dek(&key, &wrapped, b"secret data").unwrap();
        // Flip a byte in the ciphertext portion (after header + nonce).
        let flip_pos = encrypted.len() - 5;
        encrypted[flip_pos] ^= 0xFF;

        let (_, nonce_ct) = parse_envelope(&encrypted).unwrap();
        let result = decrypt_with_key(&key, nonce_ct);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("authentication failed"),
            "should detect tampered data"
        );
    }

    #[test]
    fn wrong_key_fails() {
        let key = test_key();
        let wrong_key = [0x99u8; 32];
        let wrapped = test_wrapped_dek();

        let encrypted = encrypt_with_dek(&key, &wrapped, b"secret").unwrap();
        let (_, nonce_ct) = parse_envelope(&encrypted).unwrap();
        let result = decrypt_with_key(&wrong_key, nonce_ct);
        assert!(result.is_err());
    }

    #[test]
    fn version_byte_validation() {
        let mut data = vec![0x02]; // wrong version
        data.extend_from_slice(&[6, 0]); // wrapped_dek_len
        data.extend_from_slice(&[0u8; 6]); // wrapped_dek
        data.extend_from_slice(&[0u8; 12]); // nonce
        data.extend_from_slice(&[0u8; 16]); // minimal ciphertext (just tag)

        let result = parse_envelope(&data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unsupported envelope version"));
    }

    #[test]
    fn envelope_format_parsing() {
        let key = test_key();
        let wrapped = vec![1, 2, 3, 4, 5];

        let encrypted = encrypt_with_dek(&key, &wrapped, b"test").unwrap();

        // Version byte
        assert_eq!(encrypted[0], ENVELOPE_VERSION);
        // Wrapped DEK length (LE u16)
        let len = u16::from_le_bytes([encrypted[1], encrypted[2]]) as usize;
        assert_eq!(len, wrapped.len());
        // Wrapped DEK content
        assert_eq!(&encrypted[3..3 + len], &wrapped[..]);
    }

    #[test]
    fn truncated_data_rejected() {
        // Empty
        assert!(parse_envelope(&[]).is_err());
        // Just version
        assert!(parse_envelope(&[ENVELOPE_VERSION]).is_err());
        // Version + len but no wrapped DEK / nonce / ciphertext
        assert!(parse_envelope(&[ENVELOPE_VERSION, 4, 0]).is_err());
    }

    use crate::mem::{MemBlob, MemBlobConfig};

    #[mz_ore::test(tokio::test)]
    #[cfg_attr(miri, ignore)] // unsupported operation: returning ready events from epoll_wait is not yet implemented
    async fn encrypted_blob_roundtrip() -> Result<(), ExternalError> {
        let mem = MemBlob::open(MemBlobConfig::new(false));
        let blob = EncryptedBlob::new_test(Arc::new(mem));

        // Initially empty.
        assert_eq!(blob.get("k0").await?, None);

        // Set a key and get it back (roundtrip through encrypt/decrypt).
        blob.set("k0", Bytes::from("hello")).await?;
        assert_eq!(
            blob.get("k0").await?.map(|s| s.into_contiguous()),
            Some(b"hello".to_vec())
        );

        // Overwrite and read back.
        blob.set("k0", Bytes::from("world")).await?;
        assert_eq!(
            blob.get("k0").await?.map(|s| s.into_contiguous()),
            Some(b"world".to_vec())
        );

        // Set another key.
        blob.set("k1", Bytes::from("test")).await?;
        assert_eq!(
            blob.get("k1").await?.map(|s| s.into_contiguous()),
            Some(b"test".to_vec())
        );

        // Delete returns Some (size of encrypted data, which is > plaintext).
        let deleted = blob.delete("k0").await?;
        assert!(deleted.is_some());
        assert_eq!(blob.get("k0").await?, None);

        // Double delete returns None.
        assert_eq!(blob.delete("k0").await?, None);

        // Empty value roundtrip.
        blob.set("empty", Bytes::from("")).await?;
        assert_eq!(
            blob.get("empty").await?.map(|s| s.into_contiguous()),
            Some(b"".to_vec())
        );

        // list_keys_and_metadata passes through.
        let mut keys = vec![];
        blob.list_keys_and_metadata("", &mut |entry| {
            keys.push(entry.key.to_string());
        })
        .await?;
        keys.sort();
        assert_eq!(keys, vec!["empty", "k1"]);

        Ok(())
    }
}
