# Atlas Bill of Materials

Atlas is a refurbished HP Z840 workstation used as a two-GPU ML and LLM host.
Everything below was read off the running machine on 2026-09-16
(`dmidecode`, `lscpu`, `lspci`, `lsblk`, `mdadm`, `nvidia-smi`) unless marked
"owner-stated" or "not readable from software".

## Platform

| Qty | Item | Part / model | Notes |
|---|---|---|---|
| 1 | Chassis and motherboard | HP Z840 Workstation, system board HP 2129, BIOS M60 v02.34 (2017-05-18) | Dual-socket LGA2011-3, C612 chipset, 7 PCIe slots, 4 internal 3.5" bays |
| 1 | Power supply | HP 1125 W (owner-stated) | Required for two 250 W GPUs plus two 135 W CPUs |
| 2 | CPU | Intel Xeon E5-2690 v3, 12 cores / 24 threads, 2.6 GHz base, 3.5 GHz turbo, 135 W TDP | 48 threads total |
| 2 | CPU cooler | HP Z840 liquid cooling heatsink module (owner-stated), HP part family 749599-001 / 635869-00x | Still reaches 99 °C at full load; cap jobs at about 20 threads |
| 16 | Memory | SK Hynix HMA42GR7MFR4N-TF, 16 GB DDR4-2133 ECC RDIMM | 256 GB total, all 16 slots filled, 8 per CPU |

## GPUs

| Qty | Item | Part / model | Notes |
|---|---|---|---|
| 2 | GPU | NVIDIA Quadro GV100, 32 GB HBM2, Volta GV100GL, 5120 CUDA cores, 640 tensor cores, 250 W, PCIe 3.0 x16 | One per CPU (bus 04:00 on CPU0, 84:00 on CPU1). NVLink is not possible on the Z840 because each CPU exposes one x16 root port, so the bridge cannot span sockets. GPU-to-GPU traffic goes over QPI. |

## Storage

| Qty | Item | Part / model | Role |
|---|---|---|---|
| 1 | NVMe SSD | Samsung 970 EVO Plus 2 TB (M.2 on a PCIe adapter card, adapter model not readable) | OS root, Ubuntu LVM, 1.8 TB |
| 2 | SATA SSD | Samsung 870 series 1 TB | `md1` RAID1 mirror, 1 TB |
| 3 | SAS HDD | HGST Ultrastar He8 HUH728080AL4200, 8 TB, 7200 rpm, SAS 12 Gb/s, 4Kn | `md5` RAID5, 16 TB usable, mounted at `/archive` |
| 2 | SATA SSD (spare) | Samsung PM853T MZ7GE960, 960 GB enterprise | Not in any active array (one carries a stale RAID superblock) |
| 1 | HBA | HP-branded LSI SAS2308 8-port SAS-2 HBA (HP subsystem 103c:158b, `mpt3sas`) | Drives the SAS HDDs and two of the SSDs |
| 1 | Optical | HL DVDRAM GUB0N slim | Unused |

Onboard Intel C612 SATA controllers (two, in RAID mode) carry the other SSDs.

## Networking and peripherals

| Qty | Item | Part / model | Notes |
|---|---|---|---|
| 2 | Ethernet (onboard) | Intel I218-LM (`eno1`, in use) and Intel I210 (`enp5s0`, unused) | Both gigabit |
| 1 | Keyboard | Logic3 G-720 | |
| 1 | Mouse | Logitech M90/M100 | |

## Software baseline

Ubuntu 24.04.2 LTS, NVIDIA driver 570.211.01, CUDA 12.8, Python 3.12 venv with
PyTorch 2.10+cu128. Tailscale for remote access. Details in
`ATLAS_OPERATIONS.md`.

## Rough used-market pricing (September 2026, unverified)

Only items with a price found on a live listing are quoted. Check current
listings before buying; these move week to week.

| Item | Observed price | Source |
|---|---|---|
| Quadro GV100 32 GB, refurbished | about $8,000 each from a refurbisher; eBay used listings are typically far lower but were not priced in the search | IT Creations listing |
| HGST He8 8 TB SAS, used | about $45 to $90 each | eBay sold listing, eBay.de listing |
| HP Z840 barebone with 2x E5-2690 v3 | listings exist on eBay, Newegg Refurbished and PCSP; prices not shown in search results | see links in the notes below |

## Notes for a student build

- The Z840 platform itself (chassis, board, PSU, two E5-2690 v3, 128 to 256 GB
  RDIMM) is cheap and plentiful used. The GV100s are the expensive and hard part.
- A GV100 is a 2018 Volta card. For the same money a used RTX A6000 (48 GB) or
  two used RTX 3090 (24 GB each) gives more VRAM per dollar and newer CUDA
  features, and both fit the Z840 with the 1125 W PSU. Verify card length and
  the 8-pin power leads before buying.
- The Z840 has no NVLink path between CPUs, so plan on single-GPU training or
  data-parallel jobs, not model-parallel jobs that depend on fast peer access.
- Thermals are the real limit. Air or liquid, the CPUs hit 99 °C under 48-thread
  load. Budget for the 1125 W PSU and expect to throttle CPU jobs.
- A PCIe M.2 adapter is needed for the NVMe boot drive; the Z840 has no M.2
  slot on the board. Boot from NVMe works with BIOS M60 v02.34.
