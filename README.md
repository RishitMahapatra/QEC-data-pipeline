# Project Setup

This folder holds everything you need to run the assignment locally:

- MinIO (object storage)
- PostgreSQL 16 + Adminer
- JupyterLab with all the Python dependencies pre-installed
- The three QEC datasets, already included — no Kaggle account or API keys needed
- Connection helpers, starter stage files, and basic tests to build on

## Before you start

You need Docker and Make installed, and a Bash-compatible terminal.

**Windows users:** PowerShell won't work with these commands. Install WSL 2 with Ubuntu, enable Docker Desktop's WSL integration, and run everything from the Ubuntu terminal instead. See the [Windows setup steps](assignment/getting-started.md#windows-use-wsl-bash-not-powershell) for details.

## Getting it running

From this folder, run:
`make bootstrap` 

The first run will take a few minutes since it's pulling container images. Once it's done, it'll print out three localhost links (JupyterLab, MinIO, Adminer).

Run `make help` any time to see other commands — checking status, resetting things, shutting down, etc.


## How to run the project

From this folder (after running `make bootstrap`):

```bash
docker compose exec workspace make run     # Part 1
docker compose exec workspace make train   # Part 2
docker compose exec workspace make test    # tests
```

Runing Part 1 twice gives identical outputs and no duplicate records. For a run from scratch: `make reset-platform`, then `make bootstrap` again and the commands above.



### Where to find everything:

- **Part I design report (start here):** [reports/part1.md](reports/part1.md) and [reports/part2.md](reports/part2.md)
- Discovery questions and evidence: [docs/decision-log.md](docs/decision-log.md)
- Pipeline code and per-stage overview: [starter/README.md](starter/README.md)
- Run evidence: `starter/results/part1/` (after `make run`)
