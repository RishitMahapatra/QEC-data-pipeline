# Project Setup

This folder holds everything you need to run the assignment locally:

- MinIO (object storage)
- PostgreSQL 16 + Adminer
- JupyterLab with all the Python dependencies pre-installed
- The three QEC datasets, already included — no Kaggle account or API keys needed
- Connection helpers, starter stage files, and basic tests to build on

## Before you start

You need Docker and Make installed, and a Bash-compatible terminal.

**Windows users:** PowerShell won't work with these commands. Install WSL 2 with Ubuntu, enable Docker Desktop's WSL integration, and run everything from the Ubuntu terminal instead. See the [Windows setup steps](#) for details.

## Getting it running

From this folder, run:
`make bootstrap`

The first run will take a few minutes since it's pulling container images. Once it's done, it'll print out three localhost links (JupyterLab, MinIO, Adminer).

Run `make help` any time to see other commands — checking status, resetting things, shutting down, etc.

## Where to go next

- Assignment instructions: `assignment/README.md`
- Your actual coding workspace: `starter/README.md`
