# Contributing to AGI-HPC

First off, thank you for your interest in contributing to AGI-HPC.  
This project explores a high-performance, safety-conscious architecture for AGI-like systems, and we care deeply about **technical rigor, safety, and responsible use**.

This document explains how to contribute code, ideas, and documentation in a way that’s aligned with the project’s goals and license.

---

## 1. Ground Rules

By contributing to this repository, you agree that:

1. Your contributions will be licensed under the  
   **AGI-HPC Responsible AI License v1.0** (see `LICENSE`).
2. You will **not** contribute code or content intended to:
   - enable harmful or malicious use,
   - weaken or bypass safety, governance, or audit mechanisms,
   - promote weapons, coercive surveillance, or rights violations.
3. You will follow the spirit of **safety-first, transparency, and accountability** in all contributions.

If you’re unsure whether an idea or feature is appropriate, please open an issue and ask first.

---

## 2. Code of Conduct (Short Version)

- Be respectful and constructive.
- Assume good faith, but be honest about risks and limitations.
- No harassment, hate speech, or discriminatory behavior.
- No advocating for harmful uses of AGI or this project.

If you see behavior that violates these principles, please contact the maintainer(s) at:

> **andrew.bond@sjsu.edu** (or updated project contact)

A more detailed code of conduct may be added later as the project grows.

---

## 3. How to Propose Changes

### 🐛 Reporting Bugs

1. Check existing issues to see if it’s already reported.
2. Open a new issue with:
   - a clear title,
   - steps to reproduce,
   - expected vs actual behavior,
   - environment details (OS, Python version, hardware, etc.).

### 💡 Suggesting Features or Design Changes

1. Open a **GitHub Issue** labeled `enhancement` or `design`.
2. Describe:
   - the problem or capability you care about,
   - how it fits into the AGI-HPC architecture,
   - any safety / governance implications you can think of.

For major changes (e.g., new module type, new safety mechanism, new memory layer), please start with a design discussion before opening a PR.

---

## 4. Development Workflow

### 4.1 Fork and Branch

1. **Fork** the repo to your own GitHub account.
2. Create a new branch for your work:
   ```bash
   git checkout -b feature/my-new-feature
