#!/usr/bin/env python3
"""Definition-level exact replay of the stable equality-accessible q4 bases.

The verifier has three independent parts.

1.  For a standard monomial transfer group of order at least six, enumerate
    all nonempty computational supports and impose the two root-pencil
    alternatives on every unordered port pair.  The surviving supports are
    compared with the closed-form parity/complement atlas.
2.  In the standard Pauli Bell basis, start with every partial-injection
    coefficient support.  For the other four port pairs, intersect the exact
    preimages of the four allowed Bell lines in every kernel context.  This
    reconstructs the four maximal parity/complement planes.
3.  Repeat the subspace calculation over Q(i) for the exotic V4 basis.  The
    resulting 24 maximal planes are checked to be one orbit under physical
    local exotic dressings and adjacent port swaps.  A representative plane
    is evaluated symbolically, including the exceptional X dressing needed
    when its original endpoints vanish.

Zero cards are never normalized away: membership in an allowed line is
implemented as a union of linear preimages, and the zero output belongs to
every branch.  The calculation uses only Fraction-based exact arithmetic;
there is no floating point, random sampling, external payload, or stored
survivor list.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from time import perf_counter
from typing import Iterable, Sequence


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


@dataclass(frozen=True, slots=True)
class GaussianQ:
    """An element ``real + imag*i`` of Q(i)."""

    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    @staticmethod
    def coerce(value: "GaussianQ | Fraction | int") -> "GaussianQ":
        if isinstance(value, GaussianQ):
            return value
        if isinstance(value, Fraction):
            return GaussianQ(value)
        return GaussianQ(Fraction(value))

    def __add__(self, other: "GaussianQ | Fraction | int") -> "GaussianQ":
        rhs = GaussianQ.coerce(other)
        return GaussianQ(self.real + rhs.real, self.imag + rhs.imag)

    __radd__ = __add__

    def __neg__(self) -> "GaussianQ":
        return GaussianQ(-self.real, -self.imag)

    def __sub__(self, other: "GaussianQ | Fraction | int") -> "GaussianQ":
        return self + (-GaussianQ.coerce(other))

    def __rsub__(self, other: "GaussianQ | Fraction | int") -> "GaussianQ":
        return GaussianQ.coerce(other) - self

    def __mul__(self, other: "GaussianQ | Fraction | int") -> "GaussianQ":
        rhs = GaussianQ.coerce(other)
        return GaussianQ(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )

    __rmul__ = __mul__

    def inverse(self) -> "GaussianQ":
        denominator = self.real * self.real + self.imag * self.imag
        if denominator == 0:
            raise ZeroDivisionError
        return GaussianQ(self.real / denominator, -self.imag / denominator)

    def __truediv__(self, other: "GaussianQ | Fraction | int") -> "GaussianQ":
        return self * GaussianQ.coerce(other).inverse()

    def __bool__(self) -> bool:
        return self.real != 0 or self.imag != 0


ZERO = GaussianQ()
ONE = GaussianQ(Fraction(1))
I = GaussianQ(Fraction(0), Fraction(1))

Scalar = GaussianQ
Vector = tuple[Scalar, ...]
Matrix = tuple[Vector, ...]
Space = tuple[Vector, ...]
Matrix2 = tuple[Scalar, Scalar, Scalar, Scalar]


def q(value: GaussianQ | Fraction | int) -> GaussianQ:
    return GaussianQ.coerce(value)


def vector(values: Iterable[GaussianQ | Fraction | int]) -> Vector:
    return tuple(q(value) for value in values)


def rref(rows: Iterable[Sequence[Scalar]], width: int) -> Space:
    """Canonical reduced row-echelon basis over Q(i)."""

    work = [list(row) for row in rows]
    assert all(len(row) == width for row in work)
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [entry / scale for entry in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [
                left - scale * right
                for left, right in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return tuple(tuple(row) for row in work[:pivot_row])


def nullspace(rows: Iterable[Sequence[Scalar]], width: int) -> Space:
    reduced = rref(rows, width)
    pivots = tuple(
        next(column for column, entry in enumerate(row) if entry)
        for row in reduced
    )
    output = []
    for free in range(width):
        if free in pivots:
            continue
        basis = [ZERO] * width
        basis[free] = ONE
        for row, pivot in zip(reduced, pivots):
            basis[pivot] = -row[free]
        output.append(tuple(basis))
    return tuple(output)


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    assert left and right and len(left[0]) == len(right)
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column]
                 for inner in range(len(right))), ZERO)
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def matrix_inverse(matrix: Matrix) -> Matrix:
    dimension = len(matrix)
    assert dimension > 0 and all(len(row) == dimension for row in matrix)
    augmented = tuple(
        tuple(matrix[row])
        + tuple(ONE if row == column else ZERO for column in range(dimension))
        for row in range(dimension)
    )
    reduced = rref(augmented, 2 * dimension)
    assert len(reduced) == dimension
    assert all(
        reduced[row][column] == (ONE if row == column else ZERO)
        for row in range(dimension)
        for column in range(dimension)
    )
    return tuple(
        tuple(reduced[row][dimension + column] for column in range(dimension))
        for row in range(dimension)
    )


def space_contained(left: Space, right: Space, width: int) -> bool:
    """Return whether rowspace(left) is contained in rowspace(right)."""

    return len(rref((*right, *left), width)) == len(right)


def maximal_subspaces(spaces: Iterable[Space], width: int) -> set[Space]:
    """Delete components contained in another component, preserving the union."""

    unique = set(spaces)
    return {
        left
        for left in unique
        if not any(
            len(right) > len(left) and space_contained(left, right, width)
            for right in unique
        )
    }


def intersect_axis(space: Space, card_map: Matrix, axis: int) -> Space:
    """Intersect a coefficient space with the preimage of one output line."""

    dimension = len(space)
    width = len(space[0])
    constraints = tuple(
        tuple(
            sum((card_map[row][column] * space[basis][column]
                 for column in range(width)), ZERO)
            for basis in range(dimension)
        )
        for row in range(4)
        if row != axis
    )
    kernel = nullspace(constraints, dimension)
    return rref(
        (
            tuple(
                sum((coefficient[basis] * space[basis][column]
                     for basis in range(dimension)), ZERO)
                for column in range(width)
            )
            for coefficient in kernel
        ),
        width,
    )


def bits4(index: int) -> tuple[int, int, int, int]:
    return tuple((index >> (3 - port)) & 1 for port in range(4))  # type: ignore[return-value]


def index4(bits: Sequence[int]) -> int:
    assert len(bits) == 4
    return sum(bits[port] << (3 - port) for port in range(4))


PORT_PAIRS = tuple(combinations(range(4), 2))
CROSS_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))


# ---------------------------------------------------------------------------
# Standard monomial support atlas


def pencil_support_allowed(left: frozenset[int],
                           right: frozenset[int]) -> bool:
    """Support form forced by a root pencil with at least three roots.

    The two coefficient matrices are either multiples of one nonsingular
    monomial matrix, or they occupy its two complementary coordinate axes.
    Residual two-bit coordinates are numbered 00,01,10,11, so complement is
    xor with 3.
    """

    if not left and not right:
        return True

    def full_monomial(support: frozenset[int]) -> bool:
        return len(support) == 2 and {entry ^ 3 for entry in support} == set(support)

    if not left:
        return full_monomial(right)
    if not right:
        return full_monomial(left)
    if left == right and full_monomial(left):
        return True
    return (
        len(left) == len(right) == 1
        and next(iter(left)) ^ next(iter(right)) == 3
    )


def monomial_safe_support(mask: int) -> bool:
    for pair in PORT_PAIRS:
        residual = tuple(port for port in range(4) if port not in pair)
        for deleted_slices in (((0, 0), (1, 1)), ((1, 0), (0, 1))):
            slice_supports = []
            for deleted in deleted_slices:
                support = set()
                for residual_word in range(4):
                    bits = [0] * 4
                    bits[pair[0]], bits[pair[1]] = deleted
                    bits[residual[0]] = residual_word >> 1
                    bits[residual[1]] = residual_word & 1
                    if mask & (1 << index4(bits)):
                        support.add(residual_word)
                slice_supports.append(frozenset(support))
            if not pencil_support_allowed(*slice_supports):
                return False
    return True


def parity_complement_masks() -> set[int]:
    output = set()
    for parity in (0, 1):
        pairs = []
        for word in range(16):
            complement = word ^ 15
            if word < complement and sum(bits4(word)) % 2 == parity:
                pairs.append((word, complement))
        assert len(pairs) == 4
        for selection in range(1, 1 << 4):
            mask = 0
            for pair_index, pair in enumerate(pairs):
                if selection & (1 << pair_index):
                    mask |= sum(1 << word for word in pair)
            output.add(mask)
    return output


def verify_monomial_support_atlas() -> Counter[int]:
    actual = {
        mask
        for mask in range(1, 1 << 16)
        if monomial_safe_support(mask)
    }
    expected = parity_complement_masks()
    assert actual == expected
    histogram = Counter(mask.bit_count() for mask in actual)
    assert histogram == Counter({2: 8, 4: 12, 6: 8, 8: 2})
    return histogram


# ---------------------------------------------------------------------------
# Common q4 subspace enumerator


def m2(values: Sequence[GaussianQ | Fraction | int]) -> Matrix2:
    assert len(values) == 4
    return tuple(q(value) for value in values)  # type: ignore[return-value]


def matrix2_entry(matrix: Matrix2, row: int, column: int) -> Scalar:
    return matrix[2 * row + column]


STANDARD_BASIS = (
    m2((1, 0, 0, 1)),
    m2((0, 1, 1, 0)),
    m2((1, 0, 0, -1)),
    m2((0, 1, -1, 0)),
)

EXOTIC_BASIS = (
    m2((1, 0, 0, 1)),
    m2((0, 1, 1, 0)),
    m2((I, 1, -1, -I)),
    m2((-I, 1, -1, I)),
)


def tensor_basis_matrix(basis: Sequence[Matrix2]) -> Matrix:
    """Columns are E_a(01) E_b(23), in computational coordinates."""

    return tuple(
        tuple(
            matrix2_entry(basis[label // 4], bits4(word)[0], bits4(word)[1])
            * matrix2_entry(basis[label % 4], bits4(word)[2], bits4(word)[3])
            for label in range(16)
        )
        for word in range(16)
    )


def card_map(
    support: Sequence[tuple[int, int]],
    pair: tuple[int, int],
    kernel: Matrix2,
    basis: Sequence[Matrix2],
    basis_inverse: Matrix,
) -> Matrix:
    """Four allowed-line coordinates of a card, column by column."""

    residual = tuple(port for port in range(4) if port not in pair)
    output = [[ZERO] * len(support) for _ in range(4)]
    for column, (left_label, right_label) in enumerate(support):
        residual_values = []
        for residual_word in range(4):
            total = ZERO
            for deleted_word in range(4):
                bits = [0] * 4
                bits[pair[0]] = deleted_word >> 1
                bits[pair[1]] = deleted_word & 1
                bits[residual[0]] = residual_word >> 1
                bits[residual[1]] = residual_word & 1
                total += (
                    matrix2_entry(basis[left_label], bits[0], bits[1])
                    * matrix2_entry(basis[right_label], bits[2], bits[3])
                    * matrix2_entry(kernel, bits[pair[0]], bits[pair[1]])
                )
            residual_values.append(total)
        coordinates = tuple(
            sum((basis_inverse[row][entry] * residual_values[entry]
                 for entry in range(4)), ZERO)
            for row in range(4)
        )
        for row in range(4):
            output[row][column] = coordinates[row]
    return tuple(tuple(row) for row in output)


def partial_injections() -> tuple[tuple[tuple[int, int], ...], ...]:
    output = [tuple()]
    for size in range(1, 5):
        for rows in combinations(range(4), size):
            for columns in combinations(range(4), size):
                for image in permutations(columns):
                    output.append(tuple(zip(rows, image)))
    assert len(output) == 209
    assert Counter(map(len, output)) == Counter({0: 1, 1: 16, 2: 72, 3: 96, 4: 24})
    return tuple(output)


PARTIAL_INJECTIONS = partial_injections()


def embedded_space(space: Space,
                   support: Sequence[tuple[int, int]]) -> Space:
    rows = []
    for row in space:
        embedded = [ZERO] * 16
        for coefficient, (left, right) in zip(row, support):
            embedded[4 * left + right] = coefficient
        rows.append(tuple(embedded))
    return rref(rows, 16)


def enumerate_deck_components(
    basis: Sequence[Matrix2],
) -> tuple[Counter[tuple[int, int]], set[Space], set[Space]]:
    """Enumerate all q4 deck-safe components from definition-level maps."""

    basis_matrix = tuple(tuple(matrix[entry] for matrix in basis)
                         for entry in range(4))
    basis_inverse = matrix_inverse(basis_matrix)
    histogram: Counter[tuple[int, int]] = Counter()
    global_components: set[Space] = set()

    for support in PARTIAL_INJECTIONS[1:]:
        width = len(support)
        identity = tuple(
            tuple(ONE if row == column else ZERO for column in range(width))
            for row in range(width)
        )
        components: set[Space] = {identity}
        for pair in CROSS_PAIRS:
            for kernel in basis:
                mapping = card_map(support, pair, kernel, basis, basis_inverse)
                branches = {
                    intersection
                    for component in components
                    for axis in range(4)
                    if (intersection := intersect_axis(component, mapping, axis))
                }
                # Removing an included component does not alter this union of
                # linear solution sets, and keeps the replay compact.
                components = maximal_subspaces(branches, width)

        for component in components:
            histogram[(width, len(component))] += 1
            global_components.add(embedded_space(component, support))

    maximal = maximal_subspaces(global_components, 16)
    return histogram, global_components, maximal


def coefficient_space_to_physical(space: Space, basis_matrix: Matrix) -> Space:
    return rref(
        (
            tuple(
                sum((basis_matrix[word][column] * row[column]
                     for column in range(16)), ZERO)
                for word in range(16)
            )
            for row in space
        ),
        16,
    )


def standard_parity_complement_planes() -> set[Space]:
    output = set()
    for parity in (0, 1):
        for sign in (1, -1):
            rows = []
            for word in range(16):
                complement = word ^ 15
                if word < complement and sum(bits4(word)) % 2 == parity:
                    row = [ZERO] * 16
                    row[word] = ONE
                    row[complement] = q(sign)
                    rows.append(tuple(row))
            assert len(rows) == 4
            output.add(rref(rows, 16))
    return output


STANDARD_COMPONENT_HISTOGRAM = Counter({
    (1, 1): 16,
    (2, 1): 96,
    (2, 2): 24,
    (3, 1): 144,
    (3, 2): 48,
    (3, 3): 16,
    (4, 1): 32,
    (4, 2): 24,
    (4, 4): 4,
})

EXOTIC_COMPONENT_HISTOGRAM = Counter({
    (1, 1): 16,
    (2, 1): 128,
    (2, 2): 8,
    (3, 1): 224,
    (3, 2): 32,
    (4, 1): 64,
    (4, 2): 32,
})


def verify_standard_pauli() -> tuple[Counter[int], set[Space]]:
    histogram, components, maximal = enumerate_deck_components(STANDARD_BASIS)
    assert histogram == STANDARD_COMPONENT_HISTOGRAM
    dimension_histogram = Counter(map(len, components))
    assert dimension_histogram == Counter({1: 16, 2: 24, 3: 16, 4: 4})
    assert Counter(map(len, maximal)) == Counter({4: 4})
    # This section stays over Q, even though the shared exact field
    # implementation is Q(i).
    assert all(not entry.imag for space in components for row in space for entry in row)

    basis_matrix = tensor_basis_matrix(STANDARD_BASIS)
    physical_maximal = {
        coefficient_space_to_physical(space, basis_matrix) for space in maximal
    }
    assert physical_maximal == standard_parity_complement_planes()
    return dimension_histogram, maximal


# ---------------------------------------------------------------------------
# Exotic V4 orbit and endpoint representative


def matrix2_multiply(left: Matrix2, right: Matrix2) -> Matrix2:
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def matrix2_transpose(matrix: Matrix2) -> Matrix2:
    return (matrix[0], matrix[2], matrix[1], matrix[3])


def frobenius(left: Matrix2, right: Matrix2) -> Scalar:
    return sum((left[index] * right[index] for index in range(4)), ZERO)


def projective_label(matrix: Matrix2, lines: Sequence[Matrix2]) -> int:
    pivot = next(index for index, entry in enumerate(matrix) if entry)
    for label, line in enumerate(lines):
        if not line[pivot]:
            continue
        scale = matrix[pivot] / line[pivot]
        if all(matrix[index] == scale * line[index] for index in range(4)):
            return label
    raise AssertionError("matrix is outside the expected projective line set")


def verify_exotic_basis_calculus() -> None:
    gram = tuple(
        tuple(frobenius(left, right) for right in EXOTIC_BASIS)
        for left in EXOTIC_BASIS
    )
    assert gram == (
        vector((2, 0, 0, 0)),
        vector((0, 2, 0, 0)),
        vector((0, 0, 0, 4)),
        vector((0, 0, 4, 0)),
    )
    assert tuple(projective_label(matrix2_transpose(matrix), EXOTIC_BASIS)
                 for matrix in EXOTIC_BASIS) == (0, 1, 3, 2)
    # Local dressings are projectively closed under fusion.
    assert {
        projective_label(matrix2_multiply(left, right), EXOTIC_BASIS)
        for left in EXOTIC_BASIS
        for right in EXOTIC_BASIS
    } == set(range(4))


def physical_local_action(matrix: Matrix2, port: int) -> Matrix:
    output = [[ZERO] * 16 for _ in range(16)]
    for target in range(16):
        target_bits = list(bits4(target))
        for source_bit in (0, 1):
            source_bits = target_bits.copy()
            source_bits[port] = source_bit
            source = index4(source_bits)
            output[target][source] = matrix2_entry(
                matrix, target_bits[port], source_bit
            )
    return tuple(tuple(row) for row in output)


def physical_swap_action(left: int, right: int) -> Matrix:
    output = [[ZERO] * 16 for _ in range(16)]
    for target in range(16):
        source_bits = list(bits4(target))
        source_bits[left], source_bits[right] = (
            source_bits[right], source_bits[left]
        )
        output[target][index4(source_bits)] = ONE
    return tuple(tuple(row) for row in output)


def conjugate_action_to_coefficients(
    physical_action: Matrix,
    basis_matrix: Matrix,
    basis_inverse: Matrix,
) -> Matrix:
    return matrix_multiply(
        matrix_multiply(basis_inverse, physical_action), basis_matrix
    )


def transform_space(space: Space, column_action: Matrix) -> Space:
    """Apply c' = column_action*c to every vector in a row-space basis."""

    return rref(
        (
            tuple(
                sum((column_action[target][source] * row[source]
                     for source in range(16)), ZERO)
                for target in range(16)
            )
            for row in space
        ),
        16,
    )


def exotic_orbit(maximal_planes: set[Space]) -> set[Space]:
    basis_matrix = tensor_basis_matrix(EXOTIC_BASIS)
    basis_inverse = matrix_inverse(basis_matrix)
    generators = []
    for port in range(4):
        for dressing in EXOTIC_BASIS[1:]:
            generators.append(conjugate_action_to_coefficients(
                physical_local_action(dressing, port),
                basis_matrix,
                basis_inverse,
            ))
    for port in range(3):
        generators.append(conjugate_action_to_coefficients(
            physical_swap_action(port, port + 1),
            basis_matrix,
            basis_inverse,
        ))
    assert len(generators) == 15

    first = [ZERO] * 16
    first[0] = first[5] = ONE
    second = [ZERO] * 16
    second[10] = second[15] = ONE
    representative = rref((tuple(first), tuple(second)), 16)
    assert representative in maximal_planes

    orbit = {representative}
    queue = deque((representative,))
    while queue:
        plane = queue.popleft()
        for generator in generators:
            image = transform_space(plane, generator)
            assert image in maximal_planes
            if image not in orbit:
                orbit.add(image)
                queue.append(image)
    assert orbit == maximal_planes
    assert len(orbit) == 24
    return orbit


def verify_exotic_raw_representative() -> None:
    basis_matrix = tensor_basis_matrix(EXOTIC_BASIS)
    first_coefficients = [ZERO] * 16
    first_coefficients[0] = first_coefficients[5] = ONE
    second_coefficients = [ZERO] * 16
    second_coefficients[10] = second_coefficients[15] = ONE

    first = tuple(
        sum((basis_matrix[word][column] * first_coefficients[column]
             for column in range(16)), ZERO)
        for word in range(16)
    )
    second = tuple(
        sum((basis_matrix[word][column] * second_coefficients[column]
             for column in range(16)), ZERO)
        for word in range(16)
    )
    minus_words = {0b0000, 0b0110, 0b1001, 0b1111}
    plus_words = {0b0011, 0b0101, 0b1010, 0b1100}
    odd_words = {word for word in range(16) if sum(bits4(word)) % 2}
    assert minus_words | plus_words | odd_words == set(range(16))
    assert all(first[word] == ONE for word in minus_words | plus_words)
    assert all(second[word] == q(-2) for word in minus_words)
    assert all(second[word] == q(2) for word in plus_words)
    assert all(first[word] == second[word] == ZERO for word in odd_words)

    # Symbolically q_{a,b}=a*first+b*second has endpoints a-2b.  If
    # a=2b != 0, dressing the last two ports by X maps both endpoints to
    # plus_words, where the value is a+2b=4b.
    x_last_two = matrix_multiply(
        physical_local_action(EXOTIC_BASIS[1], 2),
        physical_local_action(EXOTIC_BASIS[1], 3),
    )
    exceptional = tuple(q(2) * first[word] + second[word] for word in range(16))
    dressed = tuple(
        sum((x_last_two[row][column] * exceptional[column]
             for column in range(16)), ZERO)
        for row in range(16)
    )
    assert exceptional[0] == exceptional[15] == ZERO
    assert dressed[0] == dressed[15] == q(4)


def verify_exotic_v4() -> tuple[Counter[int], set[Space]]:
    verify_exotic_basis_calculus()
    histogram, components, maximal = enumerate_deck_components(EXOTIC_BASIS)
    assert histogram == EXOTIC_COMPONENT_HISTOGRAM
    dimension_histogram = Counter(map(len, components))
    assert dimension_histogram == Counter({1: 16, 2: 24})
    assert Counter(map(len, maximal)) == Counter({2: 24})
    orbit = exotic_orbit(maximal)
    verify_exotic_raw_representative()
    return dimension_histogram, orbit


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = perf_counter()
    print("Stable equality-accessible q4 exact certificate replay", flush=True)

    section = perf_counter()
    monomial_histogram = verify_monomial_support_atlas()
    print(
        "  standard monomial support atlas: "
        f"{dict(sorted(monomial_histogram.items()))}, 30 total "
        f"({perf_counter() - section:.3f}s)",
        flush=True,
    )

    section = perf_counter()
    pauli_dimensions, pauli_maximal = verify_standard_pauli()
    print(
        "  standard Pauli components: "
        f"{dict(sorted(pauli_dimensions.items()))}; "
        f"{len(pauli_maximal)} maximal parity/complement planes "
        f"({perf_counter() - section:.3f}s)",
        flush=True,
    )

    section = perf_counter()
    exotic_dimensions, exotic_planes = verify_exotic_v4()
    print(
        "  exotic V4 components: "
        f"{dict(sorted(exotic_dimensions.items()))}; "
        f"{len(exotic_planes)} maximal planes in one physical orbit; "
        "endpoint dressing verified "
        f"({perf_counter() - section:.3f}s)",
        flush=True,
    )
    print(
        f"STABLE EQUALITY-ACCESSIBLE Q4 EXACT CERTIFICATE: PASS "
        f"({perf_counter() - start:.3f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
