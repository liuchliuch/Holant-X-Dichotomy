#!/usr/bin/env python3
"""Exact support endpoint for full-V4 five-spread localization.

For a five-spread-safe live mask D, a scalar physical column of the common
Pauli-orbit code is a function in E_D: it satisfies the four-point equation
on every full live spread coset.  This verifier reconstructs all 15 mask
orbits and checks the two finite facts used by the physical support lemma.

* If D is not affine, at most four scalar functions (modulo constants) can
  have affine zero- and one-fibres for every pairwise difference.
* If D is affine, every function in E_D is Boolean quadratic.  Every
  nonaffine member has polar rank two or four.  Rank two has exactly three
  nonzero affine splitting forms and rank four has none, so a pairwise-safe
  family of affine offsets again has size at most four.

All masks, automorphisms, functions, fibres, ANFs, and ranks are generated
from the five spread spaces.  There is no random sample, digest, or loaded
survivor table.
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


def theta(x: int) -> int:
    d, ell = x & 1, (x >> 1) & 1
    return ell | ((d ^ ell) << 1)


P = tuple(range(4))
SPREAD = (
    frozenset(P),
    frozenset(t << 2 for t in P),
    frozenset(s | (s << 2) for s in P),
    frozenset(s | (theta(s) << 2) for s in P),
    frozenset(s | (theta(theta(s)) << 2) for s in P),
)


def binary_rank(rows) -> int:
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


COSETS = []
for space in SPREAD:
    family = {frozenset(offset ^ point for point in space) for offset in POINTS}
    assert len(family) == 4
    COSETS.extend(sorted(family, key=lambda coset: tuple(sorted(coset))))
assert len(COSETS) == 20


def is_affine_set(points) -> bool:
    points = set(points)
    if not points:
        return True
    origin = min(points)
    direction = {point ^ origin for point in points}
    return all(left ^ right in direction for left in direction for right in direction)


def is_spread_safe(mask: int) -> bool:
    domain = {point for point in POINTS if (mask >> point) & 1}
    return all(len(domain & coset) != 3 for coset in COSETS)


def xor_linear_map(columns: tuple[int, int, int, int]) -> tuple[int, ...]:
    output = []
    for point in POINTS:
        image = 0
        for index, column in enumerate(columns):
            if (point >> index) & 1:
                image ^= column
        output.append(image)
    return tuple(output)


spread_set = set(SPREAD)
linear_automorphisms = []
for columns in product(range(1, 16), repeat=4):
    if binary_rank(columns) != 4:
        continue
    mapping = xor_linear_map(columns)
    if {frozenset(mapping[x] for x in space) for space in SPREAD} == spread_set:
        linear_automorphisms.append(mapping)
assert len(linear_automorphisms) == 360

affine_automorphisms = tuple(
    tuple(mapping[point] ^ offset for point in POINTS)
    for mapping in linear_automorphisms
    for offset in POINTS
)


def act_mask(mask: int, permutation: tuple[int, ...]) -> int:
    output = 0
    for point in POINTS:
        if (mask >> point) & 1:
            output |= 1 << permutation[point]
    return output


safe_masks = {mask for mask in range(1 << 16) if is_spread_safe(mask)}
unseen = set(safe_masks)
orbits = []
while unseen:
    representative = min(unseen)
    orbit = {act_mask(representative, permutation)
             for permutation in affine_automorphisms}
    unseen -= orbit
    orbits.append((representative, orbit))

EXPECTED_ORBITS = (
    (0x0000, 0, 1),
    (0x0001, 1, 16),
    (0x0003, 2, 120),
    (0x000F, 4, 20),
    (0x0013, 3, 480),
    (0x001F, 5, 240),
    (0x0033, 4, 120),
    (0x0035, 4, 720),
    (0x003F, 6, 360),
    (0x00FF, 8, 30),
    (0x0136, 5, 288),
    (0x0365, 6, 48),
    (0x111F, 7, 160),
    (0x359F, 10, 48),
    (0xFFFF, 16, 1),
)
assert tuple((mask, mask.bit_count(), len(orbit)) for mask, orbit in orbits) == EXPECTED_ORBITS


def solution_functions(mask: int):
    domain = tuple(point for point in POINTS if (mask >> point) & 1)
    local = {point: index for index, point in enumerate(domain)}
    equations = [
        sum(1 << local[point] for point in coset)
        for coset in COSETS
        if coset <= set(domain)
    ]
    functions = tuple(
        function for function in range(1 << len(domain))
        if all((function & equation).bit_count() % 2 == 0
               for equation in equations)
    )
    return domain, functions


def maximum_clique(adjacency: tuple[int, ...]) -> tuple[int, ...]:
    best: tuple[int, ...] = ()

    def search(chosen: tuple[int, ...], candidates: int) -> None:
        nonlocal best
        if len(chosen) + candidates.bit_count() <= len(best):
            return
        if not candidates:
            if len(chosen) > len(best):
                best = chosen
            return
        vertices = [index for index in range(len(adjacency))
                    if (candidates >> index) & 1]
        pivot = max(vertices,
                    key=lambda index: (adjacency[index] & candidates).bit_count())
        branch = candidates & ~adjacency[pivot]
        while branch:
            bit = branch & -branch
            vertex = bit.bit_length() - 1
            search(chosen + (vertex,), candidates & adjacency[vertex])
            candidates &= ~bit
            branch &= ~bit

    search((), (1 << len(adjacency)) - 1)
    return best


NONAFFINE_CLIQUES = {}
for mask, _ in orbits:
    if not mask:
        continue
    domain, functions = solution_functions(mask)
    if is_affine_set(domain):
        continue
    # Quotient complementary functions by fixing their value at the first
    # domain point.  Equal normalized functions are unsafe duplicates because
    # their difference has the full nonaffine domain as one fibre.
    functions = tuple(function for function in functions if not (function & 1))
    adjacency = [0] * len(functions)
    for left, f_left in enumerate(functions):
        for right, f_right in enumerate(functions):
            if left == right:
                continue
            difference = f_left ^ f_right
            zero_fibre = {
                point for index, point in enumerate(domain)
                if not ((difference >> index) & 1)
            }
            one_fibre = set(domain) - zero_fibre
            if is_affine_set(zero_fibre) and is_affine_set(one_fibre):
                adjacency[left] |= 1 << right
    clique = maximum_clique(tuple(adjacency))
    assert len(clique) <= 4
    NONAFFINE_CLIQUES[mask] = len(clique)

assert NONAFFINE_CLIQUES == {
    0x0013: 4,
    0x001F: 2,
    0x0035: 4,
    0x003F: 4,
    0x0136: 1,
    0x0365: 1,
    0x111F: 1,
    0x359F: 1,
}


def affine_coordinates(domain: tuple[int, ...]):
    origin = domain[0]
    basis = []
    for point in domain[1:]:
        vector = point ^ origin
        if binary_rank((*basis, vector)) > len(basis):
            basis.append(vector)
    coordinate = {}
    for word in range(1 << len(basis)):
        point = origin
        for index, vector in enumerate(basis):
            if (word >> index) & 1:
                point ^= vector
        coordinate[point] = word
    assert len(coordinate) == len(domain)
    return coordinate, len(basis)


def mobius(values: list[int], dimension: int) -> list[int]:
    coefficients = values[:]
    for bit in range(dimension):
        for word in range(1 << dimension):
            if (word >> bit) & 1:
                coefficients[word] ^= coefficients[word ^ (1 << bit)]
    return coefficients


AFFINE_POLAR_DATA = {}
for mask, _ in orbits:
    if not mask:
        continue
    domain, functions = solution_functions(mask)
    if not is_affine_set(domain):
        continue
    coordinate, dimension = affine_coordinates(domain)
    rank_histogram = Counter()
    splitter_histogram = Counter()
    for function in functions:
        values = [0] * (1 << dimension)
        for index, point in enumerate(domain):
            values[coordinate[point]] = (function >> index) & 1
        anf = mobius(values, dimension)
        degree = max((monomial.bit_count()
                      for monomial, coefficient in enumerate(anf)
                      if coefficient), default=-1)
        assert degree <= 2
        if degree <= 1:
            continue
        polar_rows = []
        for left in range(dimension):
            row = 0
            for right in range(dimension):
                if left != right and anf[(1 << left) | (1 << right)]:
                    row |= 1 << right
            polar_rows.append(row)
        polar_rank = binary_rank(polar_rows)
        rank_histogram[polar_rank] += 1

        splitting_forms = 0
        for linear_form in range(1, 1 << dimension):
            kernel = [word for word in range(1 << dimension)
                      if (word & linear_form).bit_count() % 2 == 0]
            # A quadratic restricts affinely to both parallel hyperplanes iff
            # its polar form vanishes on the common kernel direction.
            def polar_image(vector: int) -> int:
                image = 0
                for bit, row in enumerate(polar_rows):
                    if (vector >> bit) & 1:
                        image ^= row
                return image

            if all(((polar_image(left) & right).bit_count() % 2) == 0
                   for left in kernel for right in kernel):
                splitting_forms += 1
        assert splitting_forms <= 3
        splitter_histogram[splitting_forms] += 1
    AFFINE_POLAR_DATA[mask] = (dict(rank_histogram), dict(splitter_histogram))

assert AFFINE_POLAR_DATA == {
    0x0001: ({}, {}),
    0x0003: ({}, {}),
    0x000F: ({}, {}),
    0x0033: ({2: 8}, {3: 8}),
    0x00FF: ({2: 48}, {3: 48}),
    0xFFFF: ({4: 96}, {0: 96}),
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("spread-safe mask orbits:", len(orbits))
    print("nonaffine-domain maximum safe cliques:")
    for mask, size in NONAFFINE_CLIQUES.items():
        print(f"  {mask:#06x}: {size}")
    print("affine-domain nonlinear polar data:")
    for mask, data in AFFINE_POLAR_DATA.items():
        print(f"  {mask:#06x}: polar-rank histogram {data[0]}, "
              f"splitting-form-count histogram {data[1]}")
    print("V4 SUPPORT-LOCALIZATION EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
