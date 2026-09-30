# Student pipeline workspace

This is the starting point for your implementation. The platform supplies the
services, dependencies, source data, configuration, connection helpers, and
simple stage templates. Your team supplies the data engineering.

Read these documents in order:

1. `../assignment/getting-started.md`
2. `../assignment/brief.md`
3. `../assignment/plain-language-guide.md`
4. `../assignment/quantum-data-primer.md`
5. `../assignment/data-sources.md`
6. [Parquet in this assignment](../assignment/plain-language-guide.md#parquet-in-this-assignment)
7. `../assignment/silver-tables.md`
8. `../assignment/required-ml-tables.md`
9. `../assignment/part-2-ai-ml.md`
10. `../assignment/rubric.md`

The starter deliberately does not contain domain transformations, target
business tables, feature builders, or fitted models. `make check` verifies
connections, `make inventory` lists the three Bronze objects, and `make test`
checks low-level utilities. `make train` still stops at an unimplemented
boundary (Part II).

## Running Part I

From a JupyterLab terminal in the workspace container (`/workspace`):

```bash
make run
```

runs every Part I stage in order with one shared `run_id`, and stops at the
first stage that fails:

| Stage | Does | Writes |
| --- | --- | --- |
| `register_sources` | verifies the unchanged Bronze objects and archive members | `results/part1/run.json`, `row_counts.json` |
| `prepare_data` | Bronze -> the six Silver Parquet tables | `silver/...`, `results/part1/source_trace.parquet`, `data_issues.parquet` |
| `load_postgres` | Silver -> Gold PostgreSQL (schema `gold`), all-or-nothing, plus the catalog and ML views | Gold tables and views |
| `build_ml_tables` | Gold views -> the two ML tables, contract checks, the three analysis queries | `ml/...`, `results/part1/analysis/query*_result.csv` |
| `trace_examples` | follows one syndrome and one Google example from ML down to the raw Bronze record | `results/part1/trace_examples.json` |

A full run takes about 30 seconds. Running it twice on unchanged input gives
identical Silver/ML files and identical Gold rows. Silver and ML live in the
MinIO bucket (`http://localhost:9001`), not in this folder.

Suggested source layout:

```text
src/quantum_lake_student/
├── config.py
├── connections.py
├── formats.py
├── ml.py
├── models.py
├── cli.py
└── stages/
    ├── register_sources.py
    ├── prepare_data.py
    ├── build_ml_tables.py
    ├── load_postgres.py
    └── train.py
```

`formats.py` provides representation-level readers for the supplied Stim `b8`
and `01` files. It does not decide how parity measurements, detector events,
shots, or decoder outputs should be modeled; those decisions remain part of
the assignment.

`ml.py` provides bit unpacking, course split assignment, the documented model
inputs, partition loading, and weighted LER. It does not select models, perform
training, or interpret results. Those remain required Part II work.

Generated data, reports, credentials, and notebook outputs should not be
committed to version control. Write required run evidence and outputs under the
`results/part1/` and `results/part2/` layout described in the brief.

The AI/ML stage is a downstream consumer check. A particular model score or an
improvement over a supplied decoder is not part of the grade.
