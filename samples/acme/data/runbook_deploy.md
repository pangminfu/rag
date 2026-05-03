# Engineering Runbook — Deploying the Fleet Console

This runbook describes the standard procedure for deploying a new release of the Acme Fleet Console to production. The Fleet Console is the customer-facing web application used to monitor and manage R-200 and X-9 fleets.

## Audience

This runbook is intended for on-call engineers in the Platform team who hold the **Deployer** role in the production AWS account. If you do not yet have this role, request it via the IT portal at least 24 hours before your first deployment.

## Prerequisites

Before deploying, confirm all of the following:

1. The release branch has been merged into `main` and tagged using semantic versioning (e.g. `v3.4.2`).
2. The CI pipeline for the tagged commit has passed — including unit tests, integration tests, and the security-scan stage.
3. The release notes have been published to the internal wiki.
4. There is no active production incident (`#incident-bridge` is empty).
5. The change has been approved in the change management system (ServiceNow CHG record linked in the PR description).

Deployment windows: **Tuesday and Thursday, 10:00–14:00 UTC.** Avoid Mondays (post-weekend issue triage) and Fridays (no on-call coverage over the weekend).

## Step-by-Step Procedure

### Step 1 — Pre-flight checks

- Run `acmectl preflight --env=prod` from your laptop. It validates that:
  - Your AWS SSO session is active.
  - The target Kubernetes context is `prod-eks-1`.
  - All required secrets exist in AWS Secrets Manager.
  - There are no in-progress migrations.
- If any check fails, stop and resolve before proceeding.

### Step 2 — Deploy to canary

- Trigger the GitHub Actions workflow `Deploy / Canary` with the release tag.
- This rolls the new image out to **5%** of the fleet-console pods behind a header-based traffic shift.
- Watch the canary dashboard at `https://grafana.acme-robotics.internal/d/canary` for **at least 15 minutes**.
- Key signals to watch:
  - HTTP 5xx rate on `/api/*` < 0.1%.
  - p95 latency within 10% of baseline.
  - No new error patterns in the canary log stream (`fleet-console-canary`).

### Step 3 — Promote to full production

- If the canary is healthy, trigger the `Deploy / Promote` workflow.
- This performs a rolling deployment to the remaining 95% of pods over approximately 10 minutes.
- Stay on the dashboard until the deployment reports 100% healthy and at least 15 minutes have passed without alerts.

### Step 4 — Run post-deploy migrations

- If the release notes flag a database migration, run:
  ```
  acmectl migrate --env=prod --release=<tag>
  ```
- Migrations are designed to be backwards-compatible with the previous version, so they may be run before, during, or after the rollout.

### Step 5 — Verify

- From an internal workstation, log into the Fleet Console and confirm:
  - The footer shows the new version tag.
  - You can view a fleet, open a device detail page, and trigger a test command.
- Post a deployment confirmation in `#deploy-log`, including the tag and any noteworthy metrics.

## Rollback

If at any point a Sev-1 or Sev-2 issue is detected, run:
```
acmectl rollback --env=prod --to=<previous-tag>
```
This reverses the rolling deployment within ~5 minutes. Do **not** roll back database migrations without consulting the on-call DBA — most migrations are forward-only by design.

## Common Issues

- **Stuck rollout:** check `kubectl rollout status deploy/fleet-console -n fleet`. If pods are crash-looping, fetch logs and roll back.
- **Secrets out of sync:** run `acmectl secrets sync --env=prod`. New env vars must be added to the secrets repo before the deploy.
- **Canary regression:** abort the workflow; the traffic shift will revert automatically within 60 seconds.

## Escalation

If you are unsure at any step, page the Platform on-call via PagerDuty service `platform-prod`. Do not "push through" — a stuck or partial deployment is much easier to resolve than a regression that has been promoted to 100% of users.
