#!/usr/bin/env python3
"""Exact phase-splitting atlas on at most four affine coordinates.

Modulo a standard stabilizer phase, only odd quadratic coefficients and
degree-at-least-three coefficients remain in a Z_4 phase polynomial.  The
script generates every such class and every nonzero linear form whose two
parallel restrictions are standard.  No random sample, digest, or loaded
table is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()


def rank(rows) -> int:
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def chart(dimension: int, linear: int, value: int) -> tuple[int, ...]:
    points = [word for word in range(1 << dimension)
              if (word & linear).bit_count() % 2 == value]
    origin = points[0]
    basis = []
    for point in points[1:]:
        vector = point ^ origin
        if rank((*basis, vector)) > len(basis):
            basis.append(vector)
    output = []
    for coordinate in range(1 << (dimension - 1)):
        point = origin
        for index, vector in enumerate(basis):
            if (coordinate >> index) & 1:
                point ^= vector
        output.append(point)
    assert set(output) == set(points)
    return tuple(output)


def mobius(values: list[int], dimension: int) -> list[int]:
    coefficients = values[:]
    for bit in range(dimension):
        for monomial in range(1 << dimension):
            if (monomial >> bit) & 1:
                coefficients[monomial] = (
                    coefficients[monomial]
                    - coefficients[monomial ^ (1 << bit)]
                ) % 4
    return coefficients


def standard(values: list[int], dimension: int) -> bool:
    for monomial, coefficient in enumerate(mobius(values, dimension)):
        degree = monomial.bit_count()
        if degree >= 3 and coefficient:
            return False
        if degree == 2 and coefficient % 2:
            return False
    return True


def evaluate(dimension: int, coefficients: dict[int, int]) -> list[int]:
    return [
        sum(coefficient for monomial, coefficient in coefficients.items()
            if word & monomial == monomial) % 4
        for word in range(1 << dimension)
    ]


def splitters(values: list[int], dimension: int) -> frozenset[int]:
    return frozenset(
        linear for linear in range(1, 1 << dimension)
        if all(standard([values[word] for word in chart(dimension, linear, value)],
                        dimension - 1)
               for value in (0, 1))
    )


CLASS_COUNTS = {}
PHASE_HISTOGRAMS = {}
SET_HISTOGRAMS = {}

for dimension in range(1, 5):
    quadratic = [monomial for monomial in range(1 << dimension)
                 if monomial.bit_count() == 2]
    higher = [monomial for monomial in range(1 << dimension)
              if monomial.bit_count() >= 3]
    counter = Counter()
    classes = 0
    for odd_mask in range(1 << len(quadratic)):
        for high_coefficients in product(range(4), repeat=len(higher)):
            if odd_mask == 0 and not any(high_coefficients):
                continue
            coefficients = {
                monomial: (odd_mask >> index) & 1
                for index, monomial in enumerate(quadratic)
            }
            coefficients.update(dict(zip(higher, high_coefficients)))
            values = evaluate(dimension, coefficients)
            assert not standard(values, dimension)
            splitting = splitters(values, dimension)
            assert len(splitting) in (0, 1, 3, 7)

            if len(splitting) == 7:
                subspace = set(splitting) | {0}
                assert rank(subspace) == 3
                assert all(left ^ right in subspace
                           for left in subspace for right in subspace)
                basis = []
                for vector in sorted(splitting):
                    if rank((*basis, vector)) > len(basis):
                        basis.append(vector)
                    if len(basis) == 3:
                        break
                cubic = [
                    2 * all((word & linear).bit_count() % 2 for linear in basis)
                    for word in range(1 << dimension)
                ]
                difference = [(left - right) % 4
                              for left, right in zip(values, cubic)]
                assert standard(difference, dimension)

            counter[splitting] += 1
            classes += 1

    CLASS_COUNTS[dimension] = classes
    PHASE_HISTOGRAMS[dimension] = Counter({
        size: sum(count for splitting, count in counter.items()
                  if len(splitting) == size)
        for size in {len(splitting) for splitting in counter}
    })
    SET_HISTOGRAMS[dimension] = Counter(len(splitting) for splitting in counter)


assert CLASS_COUNTS == {1: 0, 2: 1, 3: 31, 4: 65535}
assert PHASE_HISTOGRAMS == {
    1: Counter(),
    2: Counter({3: 1}),
    3: Counter({0: 16, 3: 14, 7: 1}),
    4: Counter({0: 64960, 1: 420, 3: 140, 7: 15}),
}
assert SET_HISTOGRAMS == {
    1: Counter(),
    2: Counter({3: 1}),
    3: Counter({3: 7, 0: 1, 7: 1}),
    4: Counter({3: 35, 7: 15, 1: 15, 0: 1}),
}


def main() -> None:
    for dimension in range(1, 5):
        print(f"dimension {dimension}: {CLASS_COUNTS[dimension]} classes, "
              f"phase classes by number of splitting forms: "
              f"{dict(sorted(PHASE_HISTOGRAMS[dimension].items()))}, "
              f"distinct splitting sets by cardinality: "
              f"{dict(sorted(SET_HISTOGRAMS[dimension].items()))}")
    print("V4 PHASE-LOCALIZATION EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
