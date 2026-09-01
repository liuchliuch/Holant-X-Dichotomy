#!/usr/bin/env python3
"""Deterministic exact replay of the A4/S4 certificates.

The program reconstructs the candidate spaces from the quaternion coordinates
specified in the paper.  It does not read precomputed payloads or use hashes,
floating-point arithmetic, random sampling, or a finite-field specialization.

Arithmetic and the basic tensor routines are shared with
``verify_a5_certificate.py``.  Its field Q(phi,i) contains Q(i), and every
number constructed here has zero phi coefficient.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from verify_a5_certificate import (
    DeckRecord,
    E,
    I,
    NATIVE_X,
    ONE,
    Q0,
    Q1,
    Q2,
    Q3,
    TWO,
    ZERO,
    Matrix2,
    Vector,
    bits4,
    contract_quaternary_pair,
    dot,
    flattening_rank,
    in_deck_cone,
    is_zero_vector,
    line_through_fixed_key,
    mat_inverse,
    mat_transpose,
    matmul,
    matrix_determinant,
    matrix_rank,
    mentry,
    product_tensor,
    projective,
    quaternion_matrix,
    solve_in_span,
    vadd,
    vscale,
    vsub,
)


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


def er(value: int | Fraction) -> E:
    return E.rational(value)


def zero_vector(length: int) -> Vector:
    return tuple(ZERO for _ in range(length))


def vector_sum(vectors: Iterable[Sequence[E]], length: int) -> Vector:
    answer = [ZERO for _ in range(length)]
    for vector in vectors:
        for index, value in enumerate(vector):
            answer[index] += value
    return tuple(answer)


def coordinate_axis(index: int) -> Vector:
    return tuple(ONE if index == j else ZERO for j in range(4))


def tetrahedral_points() -> list[Vector]:
    points = [coordinate_axis(j) for j in range(4)]
    points.extend(
        projective((ONE, er(a), er(b), er(c)))
        for a, b, c in itertools.product((-1, 1), repeat=3)
    )
    assert len(points) == len(set(points)) == 12
    return points


def octahedral_points() -> list[Vector]:
    points = tetrahedral_points()
    for left, right in itertools.combinations(range(4), 2):
        for sign in (-1, 1):
            point = [ZERO, ZERO, ZERO, ZERO]
            point[left] = ONE
            point[right] = er(sign)
            points.append(projective(tuple(point)))
    assert len(points) == len(set(points)) == 24
    return points


def verify_group(points: Sequence[Vector], expected_order: int) -> list[Matrix2]:
    matrices = [quaternion_matrix(point) for point in points]
    index = {projective(matrix) for matrix in matrices}
    assert len(index) == expected_order
    for matrix in matrices:
        assert projective(mat_inverse(matrix)) in index
        assert projective(mat_transpose(matrix)) in index
        for right in matrices:
            assert projective(matmul(matrix, right)) in index
    return matrices


def line_key(left: Sequence[E], right: Sequence[E]) -> Vector:
    """Pluecker coordinates of a projective line in P^(n-1)."""
    assert len(left) == len(right)
    return projective(tuple(
        left[i] * right[j] - left[j] * right[i]
        for i in range(len(left))
        for j in range(i + 1, len(left))
    ))


def all_secant_lines(points: Sequence[Vector]) -> dict[Vector, tuple[int, ...]]:
    pairs: dict[Vector, set[int]] = defaultdict(set)
    for left, right in itertools.combinations(range(len(points)), 2):
        key = line_key(points[left], points[right])
        pairs[key].update((left, right))
    return {key: tuple(sorted(indices)) for key, indices in pairs.items()}


def domain_line_histogram(points: Sequence[Vector]) -> Counter[int]:
    return Counter(
        len(indices) for indices in all_secant_lines(points).values()
        if len(indices) >= 3
    )


def generate_deck(points: Sequence[Vector]) -> tuple[list[DeckRecord], dict[Vector, int]]:
    matrices = [quaternion_matrix(point) for point in points]
    records: list[DeckRecord] = []
    index: dict[Vector, int] = {}
    for matching in range(3):
        for left, left_matrix in enumerate(matrices):
            for right, right_matrix in enumerate(matrices):
                vector = projective(product_tensor(matching, left_matrix, right_matrix))
                assert vector not in index
                index[vector] = len(records)
                records.append(DeckRecord(matching, left, right, vector))
    assert len(records) == len(index) == 3 * len(points) ** 2
    return records, index


def rich_lines_memory_bounded(deck: Sequence[DeckRecord]) -> list[tuple[int, ...]]:
    """Enumerate every maximal rich line with O(|deck|) live memory.

    At the least indexed point of a line, all other points occur in one
    quotient-direction bucket.  A pair cache suppresses the smaller subsets
    seen later on a four- or five-point line.
    """
    rich: list[tuple[int, ...]] = []
    covered_pairs: set[tuple[int, int]] = set()
    fast_vectors = [gaussian_vector(record.vector) for record in deck]
    for fixed_index, fixed_record in enumerate(deck):
        buckets: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
        for other_index in range(fixed_index + 1, len(deck)):
            key = gaussian_line_through_fixed_key(
                fast_vectors[fixed_index], fast_vectors[other_index]
            )
            buckets[key].append(other_index)
        for others in buckets.values():
            if len(others) < 2:
                continue
            if (fixed_index, others[0]) in covered_pairs:
                continue
            line = (fixed_index, *others)
            rich.append(line)
            for left, right in itertools.combinations(line, 2):
                covered_pairs.add((left, right))
    return rich


GaussianQ = tuple[Fraction, Fraction]
GaussianZ = tuple[int, int]


def gaussian_vector(vector: Sequence[E]) -> tuple[GaussianQ, ...]:
    answer = []
    for value in vector:
        assert value.b == value.d == 0
        answer.append((value.a, value.c))
    return tuple(answer)


def gq_mul(left: GaussianQ, right: GaussianQ) -> GaussianQ:
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def gq_sub(left: GaussianQ, right: GaussianQ) -> GaussianQ:
    return left[0] - right[0], left[1] - right[1]


def nearest_integer(value: Fraction) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    return floor + (remainder > Fraction(1, 2))


def gz_mul(left: GaussianZ, right: GaussianZ) -> GaussianZ:
    return (left[0] * right[0] - left[1] * right[1],
            left[0] * right[1] + left[1] * right[0])


def gz_divmod(left: GaussianZ, right: GaussianZ) -> tuple[GaussianZ, GaussianZ]:
    norm = right[0] * right[0] + right[1] * right[1]
    if norm == 0:
        raise ZeroDivisionError
    real = Fraction(left[0] * right[0] + left[1] * right[1], norm)
    imag = Fraction(left[1] * right[0] - left[0] * right[1], norm)
    quotient = nearest_integer(real), nearest_integer(imag)
    product = gz_mul(quotient, right)
    return quotient, (left[0] - product[0], left[1] - product[1])


def gz_gcd(left: GaussianZ, right: GaussianZ) -> GaussianZ:
    while right != (0, 0):
        _, remainder = gz_divmod(left, right)
        left, right = right, remainder
    return left


def gz_exact_div(left: GaussianZ, right: GaussianZ) -> GaussianZ:
    quotient, remainder = gz_divmod(left, right)
    assert remainder == (0, 0)
    return quotient


def canonical_gaussian_vector(vector: Sequence[GaussianQ]
                              ) -> tuple[GaussianZ, ...]:
    denominator = 1
    for real, imag in vector:
        denominator = math.lcm(denominator, real.denominator, imag.denominator)
    integers = [(int(real * denominator), int(imag * denominator))
                for real, imag in vector]
    divisor = (0, 0)
    for value in integers:
        if value != (0, 0):
            divisor = value if divisor == (0, 0) else gz_gcd(divisor, value)
    assert divisor != (0, 0)
    primitive = tuple(gz_exact_div(value, divisor) for value in integers)
    associates = []
    for unit in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        associates.append(tuple(gz_mul(unit, value) for value in primitive))
    return min(associates)


def gaussian_line_through_fixed_key(
    fixed: Sequence[GaussianQ], point: Sequence[GaussianQ]
) -> tuple[GaussianZ, ...]:
    pivot = next(index for index, value in enumerate(fixed)
                 if value != (Fraction(0), Fraction(0)))
    residual = [
        gq_sub(gq_mul(point[index], fixed[pivot]),
               gq_mul(fixed[index], point[pivot]))
        for index in range(len(fixed))
    ]
    return canonical_gaussian_vector(residual)


def matching_type(line: Sequence[int], deck: Sequence[DeckRecord]) -> tuple[int, ...]:
    counts = Counter(deck[index].matching for index in line)
    return tuple(sorted(counts.values(), reverse=True))


def lines_through_fixed(deck: Sequence[DeckRecord], fixed_index: int
                        ) -> tuple[dict[Vector, list[int]], Counter[int]]:
    fixed = deck[fixed_index].vector
    buckets: dict[Vector, list[int]] = defaultdict(list)
    for index, record in enumerate(deck):
        if index == fixed_index:
            continue
        buckets[line_through_fixed_key(fixed, record.vector)].append(index)
    return buckets, Counter(len(value) for value in buckets.values())


def enumerate_three_test_bridge(
    deck: Sequence[DeckRecord],
    deck_index: dict[Vector, int],
    a_index: int,
    b: Vector,
) -> set[Vector]:
    """Solve X,A-2X,2X-B in {0} union cone(deck), exhaustively.

    The nondependent solutions lie on a rich deck line through A.  The only
    dependent case has X and A-2X in direction A and is checked on the
    complete deck intersection with Span(A,B).
    """
    a = deck[a_index].vector
    buckets, _ = lines_through_fixed(deck, a_index)
    candidates = {zero_vector(16), vscale(Fraction(1, 2), a)}
    for other_indices in buckets.values():
        if len(other_indices) < 2:
            continue
        full_line = [a_index, *other_indices]
        for p_index, u_index in itertools.permutations(full_line, 2):
            p = deck[p_index].vector
            u = deck[u_index].vector
            coefficient_p, coefficient_u = solve_in_span(p, u, a)
            if coefficient_p.is_zero() or coefficient_u.is_zero():
                continue
            x = vscale(coefficient_p / TWO, p)
            if (in_deck_cone(x, deck_index)
                    and in_deck_cone(vsub(a, vscale(TWO, x)), deck_index)
                    and in_deck_cone(vsub(vscale(TWO, x), b), deck_index)):
                candidates.add(x)

    ab_key = line_through_fixed_key(a, projective(b))
    ab_members = [a_index, *buckets[ab_key]]
    for w_index in ab_members:
        if w_index == a_index:
            continue
        coefficient_a, _ = solve_in_span(a, deck[w_index].vector, b)
        x = vscale(coefficient_a / TWO, a)
        if (in_deck_cone(x, deck_index)
                and in_deck_cone(vsub(a, vscale(TWO, x)), deck_index)
                and in_deck_cone(vsub(vscale(TWO, x), b), deck_index)):
            candidates.add(x)
    return candidates


def card_map_rank(columns: Sequence[Vector]) -> int:
    return matrix_rank([[columns[column][row] for column in range(4)]
                        for row in range(16)])


def map_value(columns: Sequence[Vector], coordinate: Sequence[E]) -> Vector:
    return vector_sum((vscale(coordinate[j], columns[j]) for j in range(4)), 16)


def safe_on_domain(columns: Sequence[Vector], points: Sequence[Vector],
                   deck_index: dict[Vector, int]) -> bool:
    return all(in_deck_cone(map_value(columns, point), deck_index)
               for point in points)


def sum_equation_solutions(bridge: Sequence[Vector], target: Vector
                           ) -> list[tuple[Vector, Vector, Vector]]:
    return [triple for triple in itertools.product(bridge, repeat=3)
            if vector_sum(triple, 16) == target]


def tensor6_from_columns(columns: Sequence[Vector]) -> Vector:
    """Reconstruct sum_mu Q^mu_(12) tensor F_mu on six ports."""
    duals = (
        vscale(Fraction(1, 2), Q0),
        vscale(Fraction(-1, 2), Q1),
        vscale(Fraction(1, 2), Q2),
        vscale(Fraction(-1, 2), Q3),
    )
    values: list[E] = []
    for index in range(64):
        bits = tuple((index >> (5 - j)) & 1 for j in range(6))
        residual_index = 8 * bits[2] + 4 * bits[3] + 2 * bits[4] + bits[5]
        values.append(sum((
            mentry(duals[mu], bits[0], bits[1]) * columns[mu][residual_index]
            for mu in range(4)
        ), ZERO))
    return tuple(values)


def contract_tensor6_pair(tensor: Vector, pair: tuple[int, int],
                          kernel: Matrix2) -> Vector:
    assert len(tensor) == 64 and 0 <= pair[0] < pair[1] < 6
    remaining = tuple(port for port in range(6) if port not in pair)
    output: list[E] = []
    for residual in itertools.product((0, 1), repeat=4):
        total = ZERO
        for internal in itertools.product((0, 1), repeat=2):
            bits = [0] * 6
            bits[pair[0]], bits[pair[1]] = internal
            for port, bit in zip(remaining, residual):
                bits[port] = bit
            index = sum(bit << (5 - port) for port, bit in enumerate(bits))
            total += tensor[index] * mentry(kernel, internal[0], internal[1])
        output.append(total)
    return tuple(output)


def q4_ranks(vector: Vector) -> tuple[int, int, int]:
    return tuple(flattening_rank(vector, pair)
                 for pair in ((0, 1), (0, 2), (0, 3)))  # type: ignore[return-value]


def support_size(vector: Sequence[E]) -> int:
    return sum(not entry.is_zero() for entry in vector)


def transfer_of_binary(binary: Matrix2) -> Matrix2:
    return matmul(binary, NATIVE_X)


def matrix_projective_coordinates(matrix: Matrix2) -> Vector:
    """Coordinates in Q0,Q1,Q2,Q3, obtained by the Frobenius dual basis."""
    a, b, c, d = matrix
    return (
        (a + d) / TWO,
        (b + c) / (TWO * I),
        (b - c) / TWO,
        (a - d) / (TWO * I),
    )


def binary_is_in_group(binary: Matrix2, group_points: set[Vector]) -> bool:
    if all(entry.is_zero() for entry in binary):
        return False
    return projective(matrix_projective_coordinates(binary)) in group_points


def local_transform_q4(vector: Vector, matrices: Sequence[Matrix2]) -> Vector:
    assert len(matrices) == 4
    output: list[E] = []
    for out_bits in itertools.product((0, 1), repeat=4):
        total = ZERO
        for in_bits in itertools.product((0, 1), repeat=4):
            index = 8 * in_bits[0] + 4 * in_bits[1] + 2 * in_bits[2] + in_bits[3]
            coefficient = vector[index]
            for port in range(4):
                coefficient *= mentry(matrices[port], out_bits[port], in_bits[port])
            total += coefficient
        output.append(total)
    return tuple(output)


def nullspace(rows: Sequence[Sequence[E]]) -> list[Vector]:
    if not rows:
        return []
    work = [list(row) for row in rows]
    row_count, column_count = len(work), len(work[0])
    pivots: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        pivot = next((row for row in range(pivot_row, row_count)
                      if not work[row][column].is_zero()), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = work[pivot_row][column].inverse()
        work[pivot_row] = [inverse * value for value in work[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or work[row][column].is_zero():
                continue
            factor = work[row][column]
            work[row] = [x - factor * y
                         for x, y in zip(work[row], work[pivot_row])]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    free = [column for column in range(column_count) if column not in pivots]
    basis: list[Vector] = []
    for free_column in free:
        vector = [ZERO for _ in range(column_count)]
        vector[free_column] = ONE
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -sum((work[row][column] * vector[column]
                                  for column in free), ZERO)
        basis.append(tuple(vector))
    return basis


def plane_section_histogram(points: Sequence[Vector]) -> Counter[int]:
    sections: dict[Vector, tuple[int, ...]] = {}
    for triple in itertools.combinations(range(len(points)), 3):
        rows = [points[index] for index in triple]
        if matrix_rank([list(row) for row in rows]) != 3:
            continue
        normals = nullspace(rows)
        assert len(normals) == 1
        normal = projective(normals[0])
        sections[normal] = tuple(index for index, point in enumerate(points)
                                 if dot(normal, point).is_zero())
    return Counter(len(section) for section in sections.values())


def center_direction_histogram(points: Sequence[Vector]) -> Counter[int]:
    secants = all_secant_lines(points)
    bases = []
    for indices in secants.values():
        bases.append((points[indices[0]], points[indices[1]]))
    centers: set[Vector] = set()
    for (u, v), (s, t) in itertools.combinations(bases, 2):
        relation_rows = [
            (u[row], v[row], -s[row], -t[row]) for row in range(4)
        ]
        relations = nullspace(relation_rows)
        if len(relations) != 1:
            continue
        a, b, _, _ = relations[0]
        center = vadd(vscale(a, u), vscale(b, v))
        if not is_zero_vector(center):
            centers.add(projective(center))
    histogram: Counter[int] = Counter()
    for center in centers:
        directions = set()
        for point in points:
            if projective(point) == center:
                continue
            directions.add(line_through_fixed_key(center, point))
        histogram[len(directions)] += 1
    return histogram


def rank_two_projection_histogram(points: Sequence[Vector]) -> Counter[tuple[int, int, int]]:
    j_signs = (ONE, -ONE, ONE, -ONE)
    histogram: Counter[tuple[int, int, int]] = Counter()
    for indices in all_secant_lines(points).values():
        w0, w1 = points[indices[0]], points[indices[1]]
        directions: set[Vector] = set()
        killed = 0
        for root in points:
            coordinate = (
                sum((j_signs[k] * w0[k] * root[k] for k in range(4)), ZERO),
                sum((j_signs[k] * w1[k] * root[k] for k in range(4)), ZERO),
            )
            if is_zero_vector(coordinate):
                killed += 1
            else:
                directions.add(projective(coordinate))
        histogram[(len(indices), len(directions), killed)] += 1
    return histogram


def incidence_automorphism_upper_bound(points: Sequence[Vector]) -> int:
    """Enumerate automorphisms of the exact line/plane incidence structure.

    Every projective automorphism of the root configuration preserves this
    finite structure, so this is an upper bound on full-rank preservers.  The
    explicit left-right/transpose generators attain the bound below.
    """
    point_count = len(points)
    line_size: dict[tuple[int, int], int] = {}
    for indices in all_secant_lines(points).values():
        for left, right in itertools.combinations(indices, 2):
            line_size[(left, right)] = len(indices)

    planes: set[frozenset[int]] = set()
    for triple in itertools.combinations(range(point_count), 3):
        rows = [points[index] for index in triple]
        if matrix_rank([list(row) for row in rows]) != 3:
            continue
        normal = projective(nullspace(rows)[0])
        planes.add(frozenset(index for index, point in enumerate(points)
                             if dot(normal, point).is_zero()))

    colors = [[0 for _ in range(point_count)] for _ in range(point_count)]
    for left in range(point_count):
        for right in range(left + 1, point_count):
            colors[left][right] = colors[right][left] = line_size[(left, right)]
    signatures = [Counter(row) for row in colors]
    domains = [[target for target in range(point_count)
                if signatures[target] == signatures[source]]
               for source in range(point_count)]

    mapping: dict[int, int] = {}
    used: set[int] = set()
    count = 0

    def search() -> None:
        nonlocal count
        if len(mapping) == point_count:
            if all(frozenset(mapping[index] for index in plane) in planes
                   for plane in planes):
                count += 1
            return
        best_source = -1
        best_candidates: list[int] | None = None
        for source in range(point_count):
            if source in mapping:
                continue
            candidates = [target for target in domains[source]
                          if target not in used
                          and all(colors[source][old_source]
                                  == colors[target][old_target]
                                  for old_source, old_target in mapping.items())]
            if not candidates:
                return
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_source, best_candidates = source, candidates
        assert best_candidates is not None
        for target in best_candidates:
            mapping[best_source] = target
            used.add(target)
            search()
            used.remove(target)
            del mapping[best_source]

    search()
    return count


def homography_count(source: Sequence[Vector], target: Sequence[Vector]) -> int:
    """Number of PGL2 maps taking one four-point set onto the other."""
    assert len(source) == len(target) == 4
    count = 0
    for permutation in itertools.permutations(range(4)):
        equations = []
        for source_index in range(3):
            x0, x1 = source[source_index]
            y0, y1 = target[permutation[source_index]]
            equations.append((y1 * x0, y1 * x1, -y0 * x0, -y0 * x1))
        solutions = nullspace(equations)
        if len(solutions) != 1:
            continue
        a, b, c, d = solutions[0]
        if (a * d - b * c).is_zero():
            continue
        x0, x1 = source[3]
        image = projective((a * x0 + b * x1, c * x0 + d * x1))
        if image == projective(target[permutation[3]]):
            count += 1
    return count


def coordinates_on_line(points: Sequence[Vector], indices: Sequence[int]) -> list[Vector]:
    left, right = points[indices[0]], points[indices[1]]
    return [projective(solve_in_span(left, right, points[index]))
            for index in indices]


@dataclass(frozen=True)
class GroupData:
    name: str
    points: list[Vector]
    matrices: list[Matrix2]
    deck: list[DeckRecord]
    deck_index: dict[Vector, int]


def build_group(name: str) -> GroupData:
    points = tetrahedral_points() if name == "A4" else octahedral_points()
    matrices = verify_group(points, len(points))
    deck, deck_index = generate_deck(points)
    return GroupData(name, points, matrices, deck, deck_index)


def named_bridge_vectors() -> tuple[Vector, Vector, Vector, dict[str, Vector]]:
    a = product_tensor(0, Q0, Q0)
    b = product_tensor(1, Q0, Q0)
    c = product_tensor(2, Q2, Q2)
    assert c == vsub(a, b)
    named = {
        "R0": vscale(Fraction(1, 2), product_tensor(2, Q0, Q0)),
        "R1": vscale(Fraction(-1, 2), product_tensor(2, Q1, Q1)),
        "R3": vscale(Fraction(-1, 2), product_tensor(2, Q3, Q3)),
        "R01+-": vscale(Fraction(1, 4), product_tensor(
            2, tuple(x + y for x, y in zip(Q0, Q1)),
            tuple(x - y for x, y in zip(Q0, Q1)))),
        "R01-+": vscale(Fraction(1, 4), product_tensor(
            2, tuple(x - y for x, y in zip(Q0, Q1)),
            tuple(x + y for x, y in zip(Q0, Q1)))),
        "R03+-": vscale(Fraction(1, 4), product_tensor(
            2, tuple(x + y for x, y in zip(Q0, Q3)),
            tuple(x - y for x, y in zip(Q0, Q3)))),
        "R03-+": vscale(Fraction(1, 4), product_tensor(
            2, tuple(x - y for x, y in zip(Q0, Q3)),
            tuple(x + y for x, y in zip(Q0, Q3)))),
        "R13++": vscale(Fraction(-1, 4), product_tensor(
            2, tuple(x + y for x, y in zip(Q1, Q3)),
            tuple(x + y for x, y in zip(Q1, Q3)))),
        "R13--": vscale(Fraction(-1, 4), product_tensor(
            2, tuple(x - y for x, y in zip(Q1, Q3)),
            tuple(x - y for x, y in zip(Q1, Q3)))),
    }
    return a, b, c, named


def verify_a4_bridge_and_orientation(data: GroupData) -> dict[str, object]:
    a, b, c, named = named_bridge_vectors()
    a_index = data.deck_index[projective(a)]
    enumerated = enumerate_three_test_bridge(data.deck, data.deck_index,
                                             a_index, b)
    expected = {
        zero_vector(16), vscale(Fraction(1, 2), a),
        vscale(Fraction(1, 2), b), named["R0"], named["R1"], named["R3"],
    }
    assert enumerated == expected
    bridge = sorted(expected, key=lambda value: tuple(x.sort_key() for x in value))
    target = vscale(Fraction(1, 2), vadd(a, b))
    triples = sum_equation_solutions(bridge, target)
    assert len(triples) == 12

    rank_histogram = Counter()
    orientation_histogram: dict[tuple[str, str, str], tuple[int, int]] = {}
    inverse_name = {value: key for key, value in named.items()
                    if key in {"R0", "R1", "R3"}}
    group_points = set(data.points)
    rejected_supports: dict[tuple[str, str, str], int] = {}
    rejected_outsiders = 0
    deficient_histograms: list[Counter[str]] = []
    deficient_uniform_outsiders = 0
    matching_total = nonmatching_total = 0
    k_out = tuple(x + y - z + w for x, y, z, w in zip(Q0, Q1, Q2, Q3))
    assert projective(matrix_projective_coordinates(k_out)) in group_points

    for triple in triples:
        columns = (vscale(Fraction(1, 2), c), *triple)
        assert safe_on_domain(columns, data.points, data.deck_index)
        rank = card_map_rank(columns)
        rank_histogram[rank] += 1
        if rank == 2:
            tensor = tensor6_from_columns(columns)
            q_deficient = contract_tensor6_pair(tensor, (0, 2), Q0)
            assert support_size(q_deficient) == 10
            assert q4_ranks(q_deficient) == (3, 3, 3)
            histogram: Counter[str] = Counter()
            for pair in itertools.combinations(range(4), 2):
                for kernel in data.matrices:
                    binary = contract_quaternary_pair(q_deficient, pair, kernel)
                    if all(entry.is_zero() for entry in binary):
                        histogram["zero"] += 1
                    elif matrix_determinant(binary).is_zero():
                        histogram["singular"] += 1
                    elif binary_is_in_group(transfer_of_binary(binary), group_points):
                        histogram["inside"] += 1
                    else:
                        histogram["outside"] += 1
            assert histogram == Counter({"outside": 60, "inside": 9,
                                         "zero": 3})
            deficient_histograms.append(histogram)
            uniform = contract_quaternary_pair(q_deficient, (0, 1), Q0)
            assert not matrix_determinant(uniform).is_zero()
            assert not binary_is_in_group(transfer_of_binary(uniform), group_points)
            deficient_uniform_outsiders += 1
            continue
        if rank != 4:
            continue
        sigma = tuple(inverse_name[value] for value in triple)
        tensor = tensor6_from_columns(columns)
        matching_cards = 0
        nonmatching_cards = 0
        for pair in itertools.combinations(range(6), 2):
            for kernel in data.matrices:
                card = contract_tensor6_pair(tensor, pair, kernel)
                if is_zero_vector(card) or projective(card) in data.deck_index:
                    matching_cards += 1
                else:
                    nonmatching_cards += 1
        orientation_histogram[sigma] = (matching_cards, nonmatching_cards)
        matching_total += matching_cards
        nonmatching_total += nonmatching_cards

        q = contract_tensor6_pair(tensor, (0, 2), Q0)
        if nonmatching_cards:
            assert q4_ranks(q) == (2, 2, 2)
            rejected_supports[sigma] = support_size(q)
            binary = contract_quaternary_pair(q, (0, 1), k_out)
            assert not matrix_determinant(binary).is_zero()
            assert not binary_is_in_group(transfer_of_binary(binary), group_points)
            rejected_outsiders += 1

    assert rank_histogram == Counter({2: 6, 4: 6})
    assert len(deficient_histograms) == deficient_uniform_outsiders == 6
    assert Counter(orientation_histogram.values()) == Counter({(180, 0): 3,
                                                                (36, 144): 3})
    assert Counter(rejected_supports.values()) == Counter({8: 1, 16: 2})
    assert rejected_outsiders == 3
    assert (matching_total, nonmatching_total) == (648, 432)
    return {
        "bridge_vectors": len(bridge),
        "ordered_maps_by_rank": dict(sorted(rank_histogram.items())),
        "six_port_cards": matching_total + nonmatching_total,
        "matching_cards": matching_total,
        "nonmatching_cards": nonmatching_total,
        "rejected_orientation_supports": dict(Counter(rejected_supports.values())),
        "uniform_outsider_count": rejected_outsiders,
        "deficient_cards_checked": 6 * 6 * 12,
        "deficient_uniform_outsiders": deficient_uniform_outsiders,
    }


def verify_s4_bridge_and_terminals(data: GroupData) -> dict[str, object]:
    a, b, c, named = named_bridge_vectors()
    bridge = [zero_vector(16), vscale(Fraction(1, 2), a),
              vscale(Fraction(1, 2), b), *named.values()]
    assert len(bridge) == len(set(bridge)) == 12
    target = vscale(Fraction(1, 2), vadd(a, b))
    triples = sum_equation_solutions(bridge, target)
    assert len(triples) == 30
    rank_histogram = Counter()
    group_points = set(data.points)
    inverse_name = {value: key for key, value in named.items()}
    frames = {
        frozenset(("R0", "R1", "R3")),
        frozenset(("R0", "R13++", "R13--")),
        frozenset(("R1", "R03+-", "R03-+")),
        frozenset(("R3", "R01+-", "R01-+")),
    }
    category = Counter()
    support_histogram = Counter()
    outsider_kernels = {
        "R13++": (1, 1, -1),
        "R13--": (1, 1, 1),
        "R01+-": (1, 1, 1),
        "R01-+": (1, 1, -1),
    }
    outsider_transfers: set[Vector] = set()
    deficient_histograms: list[Counter[str]] = []
    deficient_uniform_outsiders = 0
    for triple in triples:
        columns = (vscale(Fraction(1, 2), c), *triple)
        # These are the candidates left by the tetrahedral sign subgroup.
        # The twelve extra octahedral pair points are precisely the separator
        # tests below, so it would be circular (and false) to assume safety on
        # all 24 points here.
        assert safe_on_domain(columns, data.points[:12], data.deck_index)
        rank = card_map_rank(columns)
        rank_histogram[rank] += 1
        if rank == 2:
            tensor = tensor6_from_columns(columns)
            q_deficient = contract_tensor6_pair(tensor, (0, 2), Q0)
            assert support_size(q_deficient) == 10
            assert q4_ranks(q_deficient) == (3, 3, 3)
            histogram: Counter[str] = Counter()
            for pair in itertools.combinations(range(4), 2):
                for kernel in data.matrices:
                    binary = contract_quaternary_pair(q_deficient, pair, kernel)
                    if all(entry.is_zero() for entry in binary):
                        histogram["zero"] += 1
                    elif matrix_determinant(binary).is_zero():
                        histogram["singular"] += 1
                    elif binary_is_in_group(transfer_of_binary(binary), group_points):
                        histogram["inside"] += 1
                    else:
                        histogram["outside"] += 1
            assert histogram == Counter({"inside": 108, "outside": 30,
                                         "zero": 6})
            deficient_histograms.append(histogram)
            uniform = contract_quaternary_pair(q_deficient, (1, 3), Q0)
            assert not matrix_determinant(uniform).is_zero()
            assert not binary_is_in_group(transfer_of_binary(uniform), group_points)
            deficient_uniform_outsiders += 1
            continue
        if rank != 4:
            continue
        names = tuple(inverse_name[value] for value in triple)
        assert frozenset(names) in frames
        q = map_value(columns, (ONE, ONE, ZERO, ZERO))  # Phi(Q0+Q1)
        first = names[0]
        support_histogram[support_size(q)] += 1
        if first in {"R0", "R3", "R03+-", "R03-+"}:
            assert q4_ranks(q) == (2, 2, 2)
            assert all(q[index].is_zero() for index in range(16)
                       if sum(bits4(index)) % 2 == 1)
            assert not q[0].is_zero() and not q[15].is_zero()
            category["raw"] += 1
        elif first == "R1":
            assert support_size(q) == 2
            category["pure_ge"] += 1
        else:
            signs = outsider_kernels[first]
            kernel = tuple(
                Q0[index] + signs[0] * Q1[index]
                + signs[1] * Q2[index] + signs[2] * Q3[index]
                for index in range(4)
            )
            assert projective(matrix_projective_coordinates(kernel)) in group_points
            binary = contract_quaternary_pair(q, (0, 1), kernel)
            assert not matrix_determinant(binary).is_zero()
            transfer = transfer_of_binary(binary)
            assert not binary_is_in_group(transfer, group_points)
            coordinates = projective(matrix_projective_coordinates(transfer))
            assert support_size(coordinates) == 3
            outsider_transfers.add(coordinates)
            category["outsider"] += 1
    assert rank_histogram == Counter({2: 6, 4: 24})
    assert len(deficient_histograms) == deficient_uniform_outsiders == 6
    assert category == Counter({"raw": 12, "pure_ge": 4, "outsider": 8})
    assert support_histogram == Counter({8: 12, 16: 8, 2: 4})
    assert len(outsider_transfers) == 4
    return {
        "bridge_vectors": len(bridge),
        "ordered_maps_by_rank": dict(sorted(rank_histogram.items())),
        "terminal_partition": dict(category),
        "distinct_outsider_transfers": len(outsider_transfers),
        "deficient_cards_checked": 6 * 6 * 24,
        "deficient_uniform_outsiders": deficient_uniform_outsiders,
    }


def verify_root_geometry(name: str, points: Sequence[Vector]) -> dict[str, object]:
    line_histogram = Counter(len(indices)
                             for indices in all_secant_lines(points).values())
    plane_histogram = plane_section_histogram(points)
    center_histogram = center_direction_histogram(points)
    projection_histogram = rank_two_projection_histogram(points)
    if name == "A4":
        assert line_histogram == Counter({2: 18, 3: 16})
        assert plane_histogram == Counter({3: 12, 6: 12})
        assert center_histogram == Counter({7: 12, 9: 12})
        assert projection_histogram == Counter({(2, 4, 2): 18,
                                                (3, 6, 0): 16})
        assert min(directions for _, directions, _ in projection_histogram) > 3
        harmonic_total = 0
    else:
        assert line_histogram == Counter({2: 72, 3: 32, 4: 18})
        assert plane_histogram == Counter({4: 96, 9: 24})
        assert center_histogram == Counter({13: 24, 19: 96})
        assert projection_histogram == Counter({(2, 6, 2): 72,
                                                (3, 6, 3): 32,
                                                (4, 4, 4): 18})
        four_lines = [indices for indices in all_secant_lines(points).values()
                      if len(indices) == 4]
        harmonic_counts = Counter()
        for source_indices in four_lines:
            source = coordinates_on_line(points, source_indices)
            for target_indices in four_lines:
                target = coordinates_on_line(points, target_indices)
                harmonic_counts[homography_count(source, target)] += 1
        assert harmonic_counts == Counter({8: 18 * 18})
        harmonic_total = 18 * 18 * 8
        assert harmonic_total == 2592
    return {
        "secant_lines": {str(key): value for key, value in sorted(line_histogram.items())},
        "plane_sections": {str(key): value for key, value in sorted(plane_histogram.items())},
        "center_directions": {str(key): value for key, value in sorted(center_histogram.items())},
        "rank_two_projection_types": {
            str(key): value for key, value in sorted(projection_histogram.items())
        },
        "rank_two_maps": harmonic_total,
    }


def verify_rank_one_and_four_generators(
    a4: GroupData, s4: GroupData,
) -> dict[str, dict[str, int]]:
    """Replay the explicit rank-one and full-rank preserver generators.

    For G=A4 the normalizer representatives are the 24 S4 points; for G=S4
    they are again the 24 S4 points.  Both left-right and transpose forms
    are generated, projectivized, and tested on A and A^sharp rootwise.
    """
    j = (ONE, -ONE, ONE, -ONE)
    answer: dict[str, dict[str, int]] = {}
    for data in (a4, s4):
        root_set = set(data.points)
        rank_one = set()
        for u in data.points:
            for v in data.points:
                jv = tuple(j[index] * v[index] for index in range(4))
                matrix = tuple(u[row] * jv[column]
                               for row in range(4) for column in range(4))
                rank_one.add(projective(matrix))
        assert len(rank_one) == len(data.points) ** 2

        full_rank = set()
        for product_uv in data.matrices:
            for v in s4.matrices:
                # The exact condition is V in the normalizer and UV in G.
                u = matmul(product_uv, mat_inverse(v))
                for transpose in (False, True):
                    columns = []
                    for basis in (Q0, Q1, Q2, Q3):
                        middle = mat_transpose(basis) if transpose else basis
                        image = matmul(matmul(u, middle), v)
                        columns.append(matrix_projective_coordinates(image))
                    matrix = tuple(columns[column][row]
                                   for row in range(4) for column in range(4))
                    full_rank.add(projective(matrix))
        expected = 2 * len(data.points) * len(s4.points)
        assert len(full_rank) == expected
        # This independently supplies completeness: any projective root-set
        # automorphism acts on the exact line/plane incidence structure, whose
        # entire automorphism group is enumerated here.  The explicit family
        # already has the same cardinality.
        assert incidence_automorphism_upper_bound(data.points) == expected

        for flattened in itertools.chain(rank_one, full_rank):
            matrix = [list(flattened[4 * row:4 * row + 4]) for row in range(4)]
            sharp = [[j[row] * matrix[column][row] * j[column]
                      for column in range(4)] for row in range(4)]
            for root in data.points:
                image = tuple(sum((matrix[row][column] * root[column]
                                   for column in range(4)), ZERO)
                              for row in range(4))
                dual_image = tuple(sum((sharp[row][column] * root[column]
                                        for column in range(4)), ZERO)
                                   for row in range(4))
                assert is_zero_vector(image) or projective(image) in root_set
                assert is_zero_vector(dual_image) or projective(dual_image) in root_set
        answer[data.name] = {
            "rank_one": len(rank_one),
            "rank_four": len(full_rank),
        }
    assert answer == {
        "A4": {"rank_one": 144, "rank_four": 576},
        "S4": {"rank_one": 576, "rank_four": 1152},
    }
    return answer


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    a4 = build_group("A4")
    s4 = build_group("S4")

    assert domain_line_histogram(a4.points) == Counter({3: 16})
    assert domain_line_histogram(s4.points) == Counter({3: 32, 4: 18})

    print("A4/S4 exact certificate replay")
    print("  groups and domains: PASS")

    a4_lines = rich_lines_memory_bounded(a4.deck)
    a4_deck_histogram = Counter((len(line), matching_type(line, a4.deck))
                                for line in a4_lines)
    assert a4_deck_histogram == Counter({(3, (1, 1, 1)): 1728,
                                        (3, (3,)): 1152})
    print("  A4 complete 432-point deck lines: PASS")

    s4_lines = rich_lines_memory_bounded(s4.deck)
    s4_deck_histogram = Counter((len(line), matching_type(line, s4.deck))
                                for line in s4_lines)
    assert s4_deck_histogram == Counter({(3, (1, 1, 1)): 13_824,
                                        (3, (3,)): 4_608,
                                        (4, (4,)): 2_592})
    print("  S4 complete 1728-point deck lines: PASS")

    a, _, _, _ = named_bridge_vectors()
    a_index = s4.deck_index[projective(a)]
    _, through_a = lines_through_fixed(s4.deck, a_index)
    assert through_a == Counter({1: 1645, 2: 32, 3: 6})
    print("  S4 all line buckets through A: PASS")

    a4_bridge = verify_a4_bridge_and_orientation(a4)
    print("  A4 bridge, 1080 six-port cards, q4 exits: PASS")
    s4_bridge = verify_s4_bridge_and_terminals(s4)
    print("  S4 bridge and first-form q4 terminal partition: PASS")

    a4_geometry = verify_root_geometry("A4", a4.points)
    s4_geometry = verify_root_geometry("S4", s4.points)
    print("  minimum-q4 secants, planes, centers, projections: PASS")

    rank_generators = verify_rank_one_and_four_generators(a4, s4)
    print("  two-sided rank-one/full-rank generators: PASS")

    print({
        "A4_bridge": a4_bridge,
        "S4_bridge": s4_bridge,
        "A4_geometry": a4_geometry,
        "S4_geometry": s4_geometry,
        "preserver_generators": rank_generators,
    })
    print("A4/S4 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
