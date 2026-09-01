#!/usr/bin/env python3
"""Run the complete exact finite-verifier suite for the companion paper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run the verifier suite with python -O."
    )


HERE = Path(__file__).resolve().parent
COMMANDS = (
    ("SC20 H6/RM8 notation dictionary", "verify_sc20_core_dictionary.py"),
    ("stable equality-accessible q4", "verify_equality_q4.py"),
    ("exotic-V4 q6/q8", "verify_exotic_v4_q6_q8.py"),
    ("affine endpoint-nondegenerate q4", "verify_affine_q4.py"),
    ("normalized-dihedral q6 support lowering", "verify_normalized_dihedral_q6.py"),
    ("full-V4 q6", "verify_v4_q6.py"),
    ("full-V4 q8 H6 branch", "verify_v4_q8_h6.py"),
    ("full-V4 five-spread", "verify_v4_spread_partial.py"),
    ("full-V4 support localization", "verify_v4_support_localization.py"),
    ("full-V4 phase localization", "verify_v4_phase_localization.py"),
    ("A4/S4 incidence and q4", "verify_a4_s4_certificates.py"),
    ("external marked A4", "verify_a4_external_form.py"),
    ("A4 hereditary q8", "verify_a4_b8.py", "--fresh"),
    ("second-form S4", "verify_s4_second_form.py"),
    ("bounded A5", "verify_a5_certificate.py"),
    ("extended Platonic incidence and H4", "verify_platonic_extended.py"),
)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for label, script, *arguments in COMMANDS:
        print(f"\n=== {label} ===", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / script), *arguments],
            cwd=HERE,
            env=environment,
            check=True,
        )
    print("\nEXACT FINITE-VERIFIER SUITE: PASS")


if __name__ == "__main__":
    main()
