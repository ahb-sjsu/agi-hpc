---
id:       tech/comms-and-networking
title:    Communications and networking
artemis:  known
sigma4:   known
topic:    tech
tags:     [tech, comms, networking, laser, encryption, light-speed]
---

# Communications and networking

Communications in 2348 face one fundamental physical
constraint that no engineering can address: the
speed of light. Within the human solar system,
communications between bodies range from negligible
delay (within a station) to *hours of one-way
delay* (between Earth and the trans-Neptunian
region). The campaign explicitly assumes this
constraint and shapes much of its narrative around
it.

Within the constraint, communications technology has
matured substantially since 2025: laser comms,
encrypted relay networks, and the substantial
inter-body and inter-station infrastructure that
makes contemporary interplanetary commerce and
politics possible.

## The physical constraint

Light-speed delay times across human space (typical;
varies with planetary positions):

| From | To | Typical one-way delay |
|---|---|---|
| Earth | Luna | 1.3 seconds |
| Earth | Mars | 4-22 minutes |
| Earth | Belt (Ceres) | 14-45 minutes |
| Earth | Jupiter system | 33-53 minutes |
| Earth | Saturn system | 70-90 minutes |
| Earth | Trans-Neptunian | 4-9 hours |
| Earth | Nithon (84 AU) | ~12 hours |

Round-trip times (which determine *interactive*
communications) are double these.

The implications:

- **Real-time conversation** is possible only
  within Earth-Luna or within a single station
- **Same-day exchanges** are possible Earth-Mars
  and Earth-Belt (with multiple round-trip
  cycles per day)
- **Multi-day exchanges** are normal Earth-outer-
  system
- **Conversations of weeks** are normal between
  Earth and trans-Neptunian (each round-trip is
  a day-plus)

Most communications planning therefore *assumes*
substantial delay. Operations build asynchronous
patterns into their architecture; conversations are
batched; decisions are pre-authorized for
foreseeable contingencies.

## Communications infrastructure

The major elements of the human-space communications
network:

### Laser comms

The primary technology for ship-to-ship and ship-to-
station communications. Properties:

- High bandwidth (multi-gigabit/sec for typical
  systems; multi-terabit/sec for high-end)
- Highly directional (point-to-point; difficult to
  intercept without being on the line)
- Vacuum-friendly (no atmospheric interference)
- Range-limited only by power and pointing accuracy

Most spacecraft have laser comms as primary; radio
backup systems are universal but secondary.

### The Outer Authority Beacon Network

Approximately 6,400 active sensor and relay beacons
distributed across the human solar system. Beacons:

- Listen passively for spacecraft signals
- Relay communications between distant points
- Serve as navigation references
- Provide environmental data (radiation, particle
  flux, etc.)

The beacon network is the largest single
communications infrastructure project in human
history. Beacons are unmanned, solar-powered, and
designed for multi-decade operational lifetimes.

### Station and ship networks

Internal networks on stations and ships are
extensive. The standard is *gigabit-class* wireless
within stations, with terabit-class wired
infrastructure for backbone links. Most personal
handhelds connect seamlessly to local networks
without explicit user action.

### Earth-Luna-Mars trunk infrastructure

The high-bandwidth communications backbone connecting
the inner planets:

- **Earth-Luna trunk** — multi-terabit/sec aggregate
  capacity; primarily fiber-equivalent for ground-
  station-to-ground-station, laser for the
  space-segment
- **Earth-Mars trunk** — substantial; uses managed
  delay protocols and multiple parallel laser
  links
- **Mars-Belt trunks** — large-scale relay through
  the trojan habitats and major Belt bodies

### Belt internal networks

The Belt's distributed-station communications are
well-developed. Most major Belt bodies have direct
laser-links to several other major bodies; the
network has substantial redundancy.

## Encryption and security

Communications encryption in 2348 has matured
substantially. The standard:

- **Quantum-key distribution (QKD)** — for high-
  security applications. The OPA, the MCR, the UN,
  and most major commercial entities use QKD for
  sensitive communications.
- **Post-quantum classical encryption** — for
  ordinary applications. Lattice-based and
  related schemes that are believed (with high
  confidence) to be resistant to quantum attack.
- **End-to-end encryption** — universal for
  consumer messaging.
- **Forward secrecy** — universal for sessions of
  any sensitivity.

The cryptographic systems are *strong*. Casual
intercept of communications is essentially
impossible. *Targeted* intercept by major intelligence
services (FSec, MIB, OPA security) is possible
through endpoint compromise rather than cryptographic
attack — i.e., they steal the keys rather than break
the cipher.

## Personal communications

Most personal communications use the standard
handheld:

- **Voice** — equivalent of 2025 phone calls;
  excellent quality
- **Text** — universal; the dominant modality
- **Video** — common; quality is high
- **Holographic presence** — for high-end
  applications; requires both ends to have
  appropriate equipment
- **Asynchronous multimedia** — voice memos, video
  messages, mixed-media exchanges

For interplanetary communications, the *delay-aware*
patterns dominate:

- *Letter-style* asynchronous messages are normal
- *Voice memos* with substantial substantive
  content are common (people *talk for ten
  minutes* and send it as a single message)
- *Conversations* are batched (each party sends
  several messages in a sequence, then waits for
  the other's batch)

## The Halyard's communications

The *Halyard*'s communications systems are typical
for a deep-range science vessel:

- **Primary**: high-power laser comms, capable of
  reaching the Outer Authority beacon network
  from outer-system distances
- **Secondary**: radio backup with substantial
  range
- **Tertiary**: emergency beacon system (the
  *EPIRB-equivalent*; legally required for
  deep-range operations)
- **Internal**: standard ship-network with
  substantial redundancy

The ship's communications relay through the Outer
Authority beacon network for trans-Neptunian
operations. Light-speed delay to Earth runs roughly
10-12 hours one-way during the *Halyard*'s
operations at Nithon.

## Communications during the campaign

Practical implications for play:

- **Real-time conversation with home is impossible**
  for the entirety of the *Halyard*'s expedition.
  Crew members send and receive *messages*, not
  *conversations*.
- **Major decisions cannot be coordinated with
  Mao-Kwikowski in real time.** Captain Marsh has
  pre-authorized authority for most operational
  decisions; novel situations would require
  *waiting for guidance* (with implied 12-24 hour
  round-trips).
- **News from home arrives delayed.** A family
  member's birthday is celebrated in real time
  on the ship; a family member's death produces
  a message that arrives hours later than the
  event.
- **The ARTEMIS/SIGMA-4 chat panels** are
  *local* — they don't round-trip to Earth. They
  respond in seconds, not hours.

## Notes for play

- **Light-speed delay** is a *narrative resource*.
  Use it. The *isolation* of the deep-range crew
  from anyone outside the ship is part of the
  campaign's psychological architecture.
- **Asynchronous messaging** can be staged in play.
  A PC might receive a video message from family
  during a quiet moment; the message is recorded,
  one-way; the PC's response will arrive hours
  later.
- **Encryption is mature.** Players cannot expect to
  intercept the Chamber's communications, the
  Caretakers' eventual communications, or
  similarly-protected channels. The campaign's
  intelligence-themes operate through other
  mechanisms.
