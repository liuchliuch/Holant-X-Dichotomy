#!/usr/bin/env python3
"""Exact five-spread replay for the partial-live full-V4 gluing atlas.

Two physical Bell labels form F_2^4 = P x P.  The five two-flat direction
spaces used here are exactly the direct simultaneous contexts:

* the horizontal and vertical Pauli-card families;
* the fixed-syndrome (s,t)=(gamma,b+gamma) R8 family;
* the two H6 triality slopes T and T^2.

Every coset has four Bell outcomes.  Its live domain is affine precisely
when it does not contain exactly three points.  The program reconstructs
all masks obeying this condition, quotients the partial masks covered by this
atlas by translations and the full linear automorphism group of the five-spread,
and performs the exact F_2 row reduction for residual stabilizer-coordinate
maps.  A full live coset imposes the parallelogram equation on every scalar
coordinate; cosets with zero, one, or two live outcomes impose none.

The calculation is definition-level finite arithmetic.  It uses no random
sampling, digest comparison, floating point, or external survivor table.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


POINTS = tuple(range(16))
FULL_MASK = (1 << 16) - 1


def theta(x: int) -> int:
    """T(d,l)=(l,d+l), with d the low bit."""
    d, ell = x & 1, (x >> 1) & 1
    return ell | ((d ^ ell) << 1)


def theta2(x: int) -> int:
    return theta(theta(x))


P = tuple(range(4))
HORIZONTAL = frozenset(P)
VERTICAL = frozenset(t << 2 for t in P)
GRAPH_I = frozenset(s | (s << 2) for s in P)
GRAPH_T = frozenset(s | (theta(s) << 2) for s in P)
GRAPH_T2 = frozenset(s | (theta2(s) << 2) for s in P)
SPREAD = (HORIZONTAL, VERTICAL, GRAPH_I, GRAPH_T, GRAPH_T2)

assert all(len(space) == 4 and 0 in space for space in SPREAD)
assert all(left & right == {0}
           for i, left in enumerate(SPREAD)
           for right in SPREAD[i + 1:])
assert set().union(*SPREAD) == set(POINTS)


def mask_of(points) -> int:
    return sum(1 << point for point in points)


COSETS = []
for space in SPREAD:
    seen = set()
    family = []
    for offset in POINTS:
        coset = frozenset(offset ^ point for point in space)
        if coset not in seen:
            seen.add(coset)
            family.append(coset)
    assert len(family) == 4
    COSETS.extend(family)
assert len(COSETS) == 20
COSET_MASKS = tuple(mask_of(coset) for coset in COSETS)


def binary_rank(rows: list[int]) -> int:
    pivots: dict[int, int] = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def affine_dimension(mask: int) -> int:
    points = [point for point in POINTS if (mask >> point) & 1]
    if not points:
        return -1
    origin = points[0]
    return binary_rank([point ^ origin for point in points[1:]])


def is_affine_mask(mask: int) -> bool:
    if not mask:
        return True
    size = mask.bit_count()
    return size == 1 << affine_dimension(mask)


def is_spread_safe(mask: int) -> bool:
    # In a four-point affine plane, every subset except a three-set is affine.
    return all((mask & coset).bit_count() != 3 for coset in COSET_MASKS)


SAFE_MASKS = tuple(mask for mask in range(1 << 16) if is_spread_safe(mask))
assert len(SAFE_MASKS) == 2652
assert Counter(mask.bit_count() for mask in SAFE_MASKS) == Counter({
    0: 1,
    1: 16,
    2: 120,
    3: 480,
    4: 860,
    5: 528,
    6: 408,
    7: 160,
    8: 30,
    10: 48,
    16: 1,
})
assert sum(is_affine_mask(mask) and mask.bit_count() == 8
           for mask in SAFE_MASKS) == 30


# Retain the partial-atlas cases by removing the empty, one-point, two-point,
# affine eight-point, and full-domain cases.
PARTIAL_MASKS = {
    mask for mask in SAFE_MASKS
    if mask.bit_count() not in (0, 1, 2, 8, 16)
}
assert len(PARTIAL_MASKS) == 2484


def linear_map(columns: tuple[int, int, int, int]) -> tuple[int, ...]:
    output = []
    for point in POINTS:
        image = 0
        for index, column in enumerate(columns):
            if (point >> index) & 1:
                image ^= column
        output.append(image)
    return tuple(output)


# GL(4,2) has 20,160 elements.  Retain exactly those permuting the five
# direction spaces; this is the complete 360-element spread automorphism
# group, not a hand-selected generator subgroup.
SPREAD_SET = set(SPREAD)
SPREAD_AUTOMORPHISMS = []
for columns in product(range(1, 16), repeat=4):
    if binary_rank(list(columns)) != 4:
        continue
    mapping = linear_map(columns)
    image_spread = {
        frozenset(mapping[point] for point in space) for space in SPREAD
    }
    if image_spread == SPREAD_SET:
        SPREAD_AUTOMORPHISMS.append(mapping)
assert len(SPREAD_AUTOMORPHISMS) == 360


AFFINE_AUTOMORPHISMS = tuple(
    tuple(mapping[point] ^ offset for point in POINTS)
    for mapping in SPREAD_AUTOMORPHISMS
    for offset in POINTS
)
assert len(AFFINE_AUTOMORPHISMS) == 5760


def act_mask(mask: int, permutation: tuple[int, ...]) -> int:
    output = 0
    while mask:
        bit = mask & -mask
        mask -= bit
        output |= 1 << permutation[bit.bit_length() - 1]
    return output


unseen = set(PARTIAL_MASKS)
ORBITS = []
while unseen:
    representative = min(unseen)
    orbit = {act_mask(representative, permutation)
             for permutation in AFFINE_AUTOMORPHISMS}
    assert orbit <= PARTIAL_MASKS
    unseen -= orbit
    ORBITS.append((representative, orbit))

EXPECTED_ORBITS = (
    (0x000F, 4, 20),
    (0x0013, 3, 480),
    (0x001F, 5, 240),
    (0x0033, 4, 120),
    (0x0035, 4, 720),
    (0x003F, 6, 360),
    (0x0136, 5, 288),
    (0x0365, 6, 48),
    (0x111F, 7, 160),
    (0x359F, 10, 48),
)
assert tuple((representative, representative.bit_count(), len(orbit))
             for representative, orbit in ORBITS) == EXPECTED_ORBITS
assert sum(len(orbit) for _, orbit in ORBITS) == 2484


def residual_dimension(mask: int) -> tuple[int, int, int]:
    """Return (affine dimension, solution nullity, nonlinear quotient)."""
    points = [point for point in POINTS if (mask >> point) & 1]
    local_index = {point: index for index, point in enumerate(points)}
    equations = []
    for coset, coset_mask in zip(COSETS, COSET_MASKS):
        if mask & coset_mask == coset_mask:
            equations.append(sum(1 << local_index[point] for point in coset))
    nullity = len(points) - binary_rank(equations)
    affine_dim = affine_dimension(mask)
    affine_restriction_dimension = affine_dim + 1
    residual = nullity - affine_restriction_dimension
    assert residual >= 0
    return affine_dim, nullity, residual


RESIDUAL_HISTOGRAM = Counter()
for mask in SAFE_MASKS:
    if not mask:
        continue
    _, _, residual = residual_dimension(mask)
    assert residual <= 2
    RESIDUAL_HISTOGRAM[residual] += 1
assert RESIDUAL_HISTOGRAM == Counter({0: 2044, 1: 576, 2: 31})


EXPECTED_REPRESENTATIVE_DATA = (
    # mask, affine-span dim, equation nullity, nonlinear residual dim
    (0x000F, 2, 3, 0),
    (0x0013, 2, 3, 0),
    (0x001F, 3, 4, 0),
    (0x0033, 2, 4, 1),
    (0x0035, 3, 4, 0),
    (0x003F, 3, 5, 1),
    (0x0136, 4, 5, 0),
    (0x0365, 4, 6, 1),
    (0x111F, 4, 5, 0),
    (0x359F, 4, 6, 1),
)
assert tuple((representative, *residual_dimension(representative))
             for representative, _ in ORBITS) == EXPECTED_REPRESENTATIVE_DATA


# On the full domain, the quotient has the expected two field-product
# coordinates.  With point bits (a,b,c,d), these are ac+bd and
# ad+bc+bd.  Together with 1,a,b,c,d they span the full seven-dimensional
# solution space.
def truth_mask(function) -> int:
    return sum(function(point) << point for point in POINTS)


ONE = FULL_MASK
COORDINATES = tuple(truth_mask(lambda point, bit=bit: (point >> bit) & 1)
                    for bit in range(4))
Q0 = truth_mask(lambda point:
                (((point >> 0) & 1) * ((point >> 2) & 1))
                ^ (((point >> 1) & 1) * ((point >> 3) & 1)))
Q1 = truth_mask(lambda point:
                (((point >> 0) & 1) * ((point >> 3) & 1))
                ^ (((point >> 1) & 1) * ((point >> 2) & 1))
                ^ (((point >> 1) & 1) * ((point >> 3) & 1)))
FULL_EQUATIONS = [coset_mask for coset_mask in COSET_MASKS]
assert binary_rank(FULL_EQUATIONS) == 9
assert binary_rank([ONE, *COORDINATES, Q0, Q1]) == 7
assert all((basis & equation).bit_count() % 2 == 0
           for basis in (ONE, *COORDINATES, Q0, Q1)
           for equation in FULL_EQUATIONS)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("five-spread safe masks:", len(SAFE_MASKS))
    print("partial masks:", len(PARTIAL_MASKS))
    print("spread automorphisms: 360 linear, 5760 affine")
    print("partial-mask orbits:", len(ORBITS))
    for representative, orbit in ORBITS:
        affine_dim, nullity, residual = residual_dimension(representative)
        print(f"  {representative:#06x}: size {representative.bit_count():2d}, "
              f"orbit {len(orbit):3d}, affine-span {affine_dim}, "
              f"coordinate-nullity {nullity}, residual {residual}")
    print("nonlinear residual histogram:", dict(sorted(RESIDUAL_HISTOGRAM.items())))
    print("V4 FIVE-SPREAD PARTIAL-LIVE EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
