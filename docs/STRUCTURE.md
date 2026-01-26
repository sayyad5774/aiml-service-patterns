# Repository Structure

This repository is organized as a pattern library for applied AIML services.

Several top-level directories were created together during initial
bootstrapping to establish a consistent layout across services.

## apps/
Contains independent AIML services (training + inference).

## docs/
Design notes and architectural context.

## infra/
Infrastructure scaffolding (local-first, cloud-mappable).

## scripts/
Developer and automation utilities.

## shared/
Shared helpers and patterns reused across services.

Early commits may touch multiple directories by design.
Later commits are scoped to specific concerns.

