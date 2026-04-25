---
id:       tech/encryption-deep
title:    Encryption — deep dive
artemis:  known
sigma4:   known
topic:    tech
tags:     [tech, encryption, cryptography, quantum, post-quantum]
---

# Encryption — deep dive

This entry covers the cryptographic foundations of
2348 in more detail than `tech/cyber-security.md`.
It is *technical* in places — relevant for PC
characters with cryptographic expertise but optional
for general players.

The summary: contemporary cryptography is *strong
enough* that ordinary cryptographic attack against
strong implementations is *not* operationally
feasible. The campaign respects this. Players who
want to *break encryption* through clever play
should expect that *strong cryptography is strong*
— operations against encrypted communications
operate through endpoint compromise, social
engineering, or other techniques rather than
cryptographic attack.

## Cryptographic foundations

### Symmetric encryption

The well-understood foundation:

- **Block ciphers** — multiple mature ciphers;
  AES-equivalent through stronger contemporary
  alternatives
- **Stream ciphers** — for specific
  applications
- **Authenticated encryption** — universal in
  modern protocols

### Key sizes

Contemporary key sizes substantially exceed
2025 standards:

- **Symmetric keys**: 256-bit minimum for
  most applications; 512-bit for high-
  security applications
- **Asymmetric keys**: post-quantum schemes
  use parameters substantially different
  from 2025 RSA/ECC; the equivalent security
  levels have been carefully analyzed

### Asymmetric encryption

The transition from pre-quantum to post-quantum
classical cryptography occurred substantially in
the late twenty-first century. Contemporary
asymmetric encryption uses:

- **Lattice-based schemes** — the dominant
  category; well-analyzed; quantum-resistant
- **Code-based schemes** — secondary; specific
  applications
- **Hash-based schemes** — for signatures
- **Isogeny-based schemes** — limited use;
  some applications

### Key exchange

Two parallel approaches:

- **Quantum-key distribution (QKD)** — physical-
  layer key exchange using quantum properties;
  the strongest security but operationally
  expensive
- **Post-quantum classical** — algorithmic
  approaches that are believed (with high
  confidence) to be quantum-resistant

QKD is used for high-security applications
(military, intelligence, major financial); post-
quantum classical is used for ordinary
applications.

### Forward secrecy

Universal in modern protocols. Forward secrecy
ensures that compromise of long-term keys does
not expose past communications.

## Specific cryptographic primitives

### Hash functions

- **SHA-3 derivatives** — substantially mature;
  the dominant choice
- **Specialty hashes** — for specific
  applications

### Authentication codes

- **HMAC-equivalents** — substantial
- **Authenticated encryption schemes** —
  combining encryption and authentication
  in single primitives

### Key derivation

- **Standard key-derivation functions** —
  substantial; used universally for
  password-based and similar applications
- **Specialty derivation functions** — for
  specific applications

### Random number generation

- **Hardware random-number generation** —
  substantial; most cryptographic systems
  use hardware RNG sources
- **Pseudo-random number generation** — for
  derived applications
- **Specialty applications** — including
  some that use specific physical processes
  for randomness

## Quantum cryptography

### Quantum-key distribution

- **Standard QKD** — for high-security
  applications
- **Long-distance QKD** — substantial but
  technically complex
- **Quantum-key distribution networks** —
  substantial deployment

### Quantum-safe vs. quantum-vulnerable

- **Quantum-safe**: lattice-based, code-based,
  hash-based, isogeny-based asymmetric
  schemes; symmetric schemes with sufficient
  key sizes
- **Quantum-vulnerable**: the pre-quantum
  asymmetric schemes (RSA, ECC) of 2025;
  these are *not* used for current
  applications

### Post-quantum migration

The migration from 2025-era cryptography to
post-quantum schemes was completed substantially
by 2090. The legacy systems still in operation
are *quarantined* and not used for sensitive
applications.

## Cryptographic applications

### Communications

- **Voice-and-text encryption** — universal
  for ordinary communications
- **Video encryption** — universal
- **Specialty applications** — including
  high-bandwidth applications, real-time
  applications, etc.

### Storage

- **At-rest encryption** — universal
- **Specialty applications** — including
  long-term-archival, redaction-resistant,
  etc.

### Authentication

- **Cryptographic identity** — universal for
  most authentication
- **Multi-factor authentication** — universal
  for sensitive applications
- **Specialty authentication** — for
  specific applications

### Specialty cryptographic applications

- **Zero-knowledge proofs** — substantial
  deployment for privacy-preserving
  applications
- **Multi-party computation** — substantial;
  for joint computation across mutually-
  distrustful parties
- **Homomorphic encryption** — substantial;
  for computation on encrypted data
- **Secure enclaves** — universal in
  modern hardware; substantial application
  for sensitive operations

## Cryptographic attacks

What *can* be done:

### Implementation attacks

- **Side-channel attacks** — through
  electromagnetic, thermal, or timing
  channels; effective against specific
  implementations
- **Fault-injection attacks** — through
  physical manipulation of cryptographic
  hardware
- **Implementation flaw exploitation** —
  through specific software/hardware flaws

### Endpoint attacks

- **Key compromise** — through endpoint
  compromise; often the most practical
  attack vector
- **Social-engineering** — for credential
  acquisition

### Cryptographic attacks (limited)

- **Specific algorithmic flaws** — rare
  but occasionally productive
- **Specific implementation flaws** — more
  common; specific applications
- **Substantial-resource attacks** — only
  feasible with nation-state-class resources
  and specific weakness exploitation

### What *can't* be done

- **Brute-force attack against properly-sized
  keys** — substantially infeasible even with
  contemporary computing
- **Generic algorithmic attack against well-
  vetted schemes** — substantially
  infeasible
- **Substantial mathematical breakthrough
  attack** — possible in principle but
  rare; would require substantial new
  mathematical insight

## Cryptography and the campaign

### What players can do

- **Use encrypted communications**
  effectively for ordinary purposes
- **Protect sensitive data** through
  appropriate cryptographic application
- **Verify authenticity** of messages and
  documents through cryptographic signatures

### What players cannot easily do

- **Break strong encryption** through
  cryptographic attack — generally not
  feasible
- **Decrypt communications** they don't
  have keys for — generally not feasible
- **Impersonate cryptographically-
  authenticated parties** — generally not
  feasible

### What players can do with skill

- **Compromise endpoints** — through
  appropriate operations (cyber, physical,
  social-engineering)
- **Exploit implementation flaws** in
  specific systems — through skill checks
  and substantial preparation
- **Use specialized cryptographic operations**
  — zero-knowledge proofs, multi-party
  computation, secure enclaves — for plot-
  relevant applications

## Notes for play

- **Strong cryptography is strong**. Players
  cannot expect to break encrypted
  communications through casual operations.
- **Endpoint compromise** is the practical
  attack vector. Operations against encrypted
  communications go through compromising the
  endpoints rather than breaking the cipher.
- **Cryptographic skills** are genuine
  technical specialties. PCs with
  cryptographic expertise have substantial
  capability in specific contexts.
- **The campaign's intelligence-themes**
  use endpoint compromise rather than
  cryptographic breaking; players should
  expect to investigate plot elements
  through human-source-and-endpoint
  techniques rather than crypto-breaking.
