# Exact finite verifiers

This directory contains the companion verification package for **A
Dichotomy for Complex Boolean Holant with Binary Disequality**. The programs
check selected finite claims underlying the paper's core dictionary;
equality-accessible and endpoint-nondegenerate quaternary analyses;
normalized-dihedral and Klein-four analyses; and Platonic matching-deck
analyses. The mathematical statements themselves are given in the paper.

The precise correspondence between manuscript anchors, finite claims, and
entry points is recorded in
[`theorem-to-script-map.md`](theorem-to-script-map.md).

## Requirements

- Python 3.10 or later.
- Python assertions must be enabled; do not use `python -O`.
- No third-party packages or network access are required.

## Run the complete suite

From the repository root, run:

```sh
python3 certificates/run_all_exact.py
```

The runner executes all 16 verifiers in a fixed order and stops at the first
failure. The complete suite is computationally intensive, and its running
time depends on the machine. In particular, the hereditary A4 eight-port
stage rebuilds its large six-port orbits in memory; on our reference run this
stage used roughly 0.5 GB of memory and several minutes. The suite does not
modify `payloads-v1/` or write a persistent orbit cache.

Each verifier may also be run directly. Keep the verifier files together:
several Platonic verifiers share exact-arithmetic definitions by importing
other files in this directory. `verify_a4_b8.py` always performs a fresh,
cache-independent reconstruction. Its `--fresh` option is retained only to
match the invocation recorded in the manuscript and used by the suite runner.

## Package layout

- `run_all_exact.py`: entry point for the complete suite.
- `verify_*.py`: individual exact verifiers.
- `theorem-to-script-map.md`: manuscript-to-verifier coverage map.
- `manifest-v1.json`: expected serialization metadata for the A5 verifier.
- `payloads-v1/`: deterministic A5 serialization snapshots.

## Trust boundary

In the complete-suite run, the programs regenerate the finite objects from
displayed definitions and use exact arithmetic. The verifiers use no network
access, random sampling, floating-point tolerances, or finite-field
specialization. They emit audit summaries and signal failure through
assertions or a nonzero exit status; they do not produce independently
checkable formal proof objects.

Reproduction still relies on the verifier implementations, the Python
interpreter and standard library, and the execution platform. The manifest
and embedded digests are regression baselines, not independent proofs.

Running this suite does not prove the analytic normalization of an arbitrary
signature, the physical availability or wiring of a gadget, factor retention
or saturation, deck stabilization, or a whole-problem reduction. Those
arguments remain in the paper.

## A5 manifest and payload snapshots

`manifest-v1.json` records schema metadata, record counts, byte lengths, and
SHA-256 digests for eight deterministic serializations:

- `group_points.jsonl`
- `domain_lines.jsonl`
- `deck_points.jsonl`
- `rich_lines_through_A.jsonl`
- `bridge_vectors.jsonl`
- `ordered_triples.jsonl`
- `frames.jsonl`
- `kernel_outputs.jsonl`

The default A5 verifier regenerates all eight streams in memory, compares
them byte-for-byte with `payloads-v1/`, and compares their metadata with the
manifest. The checked-in JSONL files are regression references: the program
does not use them to construct the finite objects or to decide which objects
survive. Pass `--no-payload-check` to replay the enumeration without reading
the reference snapshots.

Because the manifest records byte-level digests, do not reformat these JSONL
files or change their line endings.

To export regenerated snapshots explicitly, run:

```sh
python3 certificates/verify_a5_certificate.py \
  --write-payloads regenerated-payloads-v1
```

This option writes files to the specified directory, replacing any same-named
payload files already there. The default verifier suite is read-only with
respect to `payloads-v1/`. The manifest detects generated-output or
serialization drift relative to the recorded metadata; it is not an
independent proof that an enumeration is complete.
