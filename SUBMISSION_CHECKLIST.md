# GitHub Repository Checklist for Paper Submission

## Before creating the repository

- [ ] Confirm whether the venue requires double-blind anonymity.
- [ ] If double-blind, use an anonymous repository or archival anonymous link.
- [ ] Remove author names, affiliations, acknowledgements, and personal paths from the public code.
- [ ] Check that no private API keys, credentials, local machine paths, or reviewer-only notes are included.
- [ ] Decide the license before making the repository public.

## Repository contents

- [x] Source code included.
- [x] Dependency file included.
- [x] Dataset/profile file included.
- [x] README with installation and reproduction commands included.
- [x] `.gitignore` included.
- [x] Citation template included.

## Suggested Git commands

```bash
git init
git add .
git commit -m "Initial release for paper submission"
git branch -M main
git remote add origin https://github.com/REPLACE-WITH-ACCOUNT/REPLACE-WITH-REPOSITORY.git
git push -u origin main
```

## Suggested repository description

Code and experiment scripts for CQ-DSM, a cyber-physical QoS-aware demand-side management simulator with MILP-based HEMS optimization and anomaly-detection-assisted control.

## Suggested paper text

The source code and experiment scripts are available at: `REPLACE-WITH-REPOSITORY-LINK`.
