#!/usr/bin/env python3
"""Rebuild and verify the bounded A5 certificate exactly.

The verifier reconstructs eight deterministic certificate streams in memory
and, by default, compares them byte-for-byte with ``payloads-v1/`` and checks
their serialization metadata and SHA-256 digests against ``manifest-v1.json``.
The streams can also be exported with ``--write-payloads``.  The implementation
uses only the Python standard library.  All algebraic arithmetic is in
Q(phi, i), where phi^2 = phi + 1; there are no floating-point comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


SCHEMA = "a5-cert-v1"
KERNEL_OUTPUT_SCHEMA = "a5-kernel-outputs-v2"
PAYLOAD_NAMES = (
    "group_points.jsonl",
    "domain_lines.jsonl",
    "deck_points.jsonl",
    "rich_lines_through_A.jsonl",
    "bridge_vectors.jsonl",
    "ordered_triples.jsonl",
    "frames.jsonl",
    "kernel_outputs.jsonl",
)


def FQ(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def qmul(a: Fraction, b: Fraction,
         c: Fraction, d: Fraction) -> tuple[Fraction, Fraction]:
    """Multiply (a+b phi)(c+d phi), where phi^2=phi+1."""
    return a * c + b * d, a * d + b * c + b * d


def qinv(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction]:
    """Invert a+b phi in Q(phi)."""
    norm = a * a + a * b - b * b
    if norm == 0:
        raise ZeroDivisionError("zero Q(phi) norm")
    return (a + b) / norm, -b / norm


@dataclass(frozen=True, slots=True)
class E:
    """(a+b phi) + i(c+d phi), with rational a,b,c,d."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)
    c: Fraction = Fraction(0)
    d: Fraction = Fraction(0)

    @staticmethod
    def rational(value: int | Fraction) -> "E":
        return E(FQ(value))

    @staticmethod
    def coerce(value: "E | int | Fraction") -> "E":
        return value if isinstance(value, E) else E.rational(value)

    def __add__(self, other: "E | int | Fraction") -> "E":
        o = E.coerce(other)
        return E(self.a + o.a, self.b + o.b,
                 self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self) -> "E":
        return E(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, other: "E | int | Fraction") -> "E":
        return self + (-E.coerce(other))

    def __rsub__(self, other: "E | int | Fraction") -> "E":
        return E.coerce(other) - self

    def __mul__(self, other: "E | int | Fraction") -> "E":
        o = E.coerce(other)
        rr0, rr1 = qmul(self.a, self.b, o.a, o.b)
        ii0, ii1 = qmul(self.c, self.d, o.c, o.d)
        ri0, ri1 = qmul(self.a, self.b, o.c, o.d)
        ir0, ir1 = qmul(self.c, self.d, o.a, o.b)
        return E(rr0 - ii0, rr1 - ii1, ri0 + ir0, ri1 + ir1)

    __rmul__ = __mul__

    def inverse(self) -> "E":
        x20, x21 = qmul(self.a, self.b, self.a, self.b)
        y20, y21 = qmul(self.c, self.d, self.c, self.d)
        n0, n1 = x20 + y20, x21 + y21
        ni0, ni1 = qinv(n0, n1)
        r0, r1 = qmul(self.a, self.b, ni0, ni1)
        i0, i1 = qmul(-self.c, -self.d, ni0, ni1)
        return E(r0, r1, i0, i1)

    def __truediv__(self, other: "E | int | Fraction") -> "E":
        return self * E.coerce(other).inverse()

    def __rtruediv__(self, other: "E | int | Fraction") -> "E":
        return E.coerce(other) / self

    def __pow__(self, exponent: int) -> "E":
        if exponent < 0:
            return (self.inverse()) ** (-exponent)
        result = ONE
        base = self
        n = exponent
        while n:
            if n & 1:
                result *= base
            base *= base
            n >>= 1
        return result

    def is_zero(self) -> bool:
        return self == ZERO

    def sort_key(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return self.a, self.b, self.c, self.d

    def to_data(self) -> list[list[int]]:
        return [[q.numerator, q.denominator]
                for q in (self.a, self.b, self.c, self.d)]


ZERO = E.rational(0)
ONE = E.rational(1)
TWO = E.rational(2)
I = E(Fraction(0), Fraction(0), Fraction(1), Fraction(0))
PHI = E(Fraction(0), Fraction(1))
TAU = PHI - ONE

Vector = tuple[E, ...]
Matrix2 = tuple[E, E, E, E]


def vadd(left: Sequence[E], right: Sequence[E]) -> Vector:
    assert len(left) == len(right)
    return tuple(a + b for a, b in zip(left, right))


def vsub(left: Sequence[E], right: Sequence[E]) -> Vector:
    assert len(left) == len(right)
    return tuple(a - b for a, b in zip(left, right))


def vscale(scalar: E | int | Fraction, vector: Sequence[E]) -> Vector:
    s = E.coerce(scalar)
    return tuple(s * x for x in vector)


def vsum(vectors: Iterable[Sequence[E]], length: int) -> Vector:
    result = tuple(ZERO for _ in range(length))
    for vector in vectors:
        result = vadd(result, vector)
    return result


def is_zero_vector(vector: Sequence[E]) -> bool:
    return all(x.is_zero() for x in vector)


def projective(vector: Sequence[E]) -> Vector:
    for entry in vector:
        if not entry.is_zero():
            return vscale(entry.inverse(), vector)
    raise ValueError("zero vector has no projective normalization")


def vector_sort_key(vector: Sequence[E]) -> tuple:
    return tuple(x.sort_key() for x in vector)


def vector_data(vector: Sequence[E]) -> list[list[list[int]]]:
    return [x.to_data() for x in vector]


def matmul(left: Matrix2, right: Matrix2) -> Matrix2:
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def mat_transpose(matrix: Matrix2) -> Matrix2:
    a, b, c, d = matrix
    return a, c, b, d


def mat_inverse(matrix: Matrix2) -> Matrix2:
    a, b, c, d = matrix
    determinant = a * d - b * c
    if determinant.is_zero():
        raise ZeroDivisionError("singular 2x2 matrix")
    return tuple(x / determinant for x in (d, -b, -c, a))  # type: ignore[return-value]


Q0: Matrix2 = (ONE, ZERO, ZERO, ONE)
Q1: Matrix2 = (ZERO, I, I, ZERO)
Q2: Matrix2 = (ZERO, ONE, -ONE, ZERO)
Q3: Matrix2 = (I, ZERO, ZERO, -I)
QBASIS = (Q0, Q1, Q2, Q3)
NATIVE_X: Matrix2 = (ZERO, ONE, ONE, ZERO)


def quaternion_matrix(coordinates: Sequence[E]) -> Matrix2:
    assert len(coordinates) == 4
    entries = []
    for position in range(4):
        entries.append(sum((coordinates[j] * QBASIS[j][position]
                            for j in range(4)), ZERO))
    return tuple(entries)  # type: ignore[return-value]


def permutation_is_even(permutation: Sequence[int]) -> bool:
    inversions = sum(permutation[i] > permutation[j]
                     for i in range(len(permutation))
                     for j in range(i + 1, len(permutation)))
    return inversions % 2 == 0


def generate_group_points() -> list[Vector]:
    axes = [tuple(ONE if i == j else ZERO for i in range(4))
            for j in range(4)]

    signs = set()
    for eps in itertools.product((-1, 1), repeat=3):
        signs.add(projective((ONE, E.rational(eps[0]),
                              E.rational(eps[1]), E.rational(eps[2]))))

    last = set()
    even_permutations = [p for p in itertools.permutations(range(4))
                         if permutation_is_even(p)]
    for eps in itertools.product((-1, 1), repeat=3):
        base = (ZERO, E.rational(eps[0]),
                E.rational(eps[1]) * PHI,
                E.rational(eps[2]) * TAU)
        for permutation in even_permutations:
            last.add(projective(tuple(base[permutation[j]] for j in range(4))))

    assert (len(axes), len(signs), len(last)) == (4, 8, 48)
    points = axes + sorted(signs, key=vector_sort_key) \
                  + sorted(last, key=vector_sort_key)
    assert len(points) == len(set(points)) == 60
    return points


def wedge_key(left: Sequence[E], right: Sequence[E]) -> Vector:
    assert len(left) == len(right)
    wedge = tuple(left[i] * right[j] - left[j] * right[i]
                  for i in range(len(left))
                  for j in range(i + 1, len(left)))
    return projective(wedge)


def domain_rich_lines(points: Sequence[Vector]) -> list[tuple[int, ...]]:
    lines: dict[Vector, set[int]] = defaultdict(set)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            key = wedge_key(points[i], points[j])
            lines[key].update((i, j))
    rich = sorted((tuple(sorted(indices)) for indices in lines.values()
                   if len(indices) >= 3))
    assert Counter(map(len, rich)) == Counter({3: 200, 5: 72})
    incidences = Counter(index for line in rich for index in line)
    for index in range(60):
        sizes = Counter(len(line) for line in rich if index in line)
        assert sizes == Counter({3: 10, 5: 6})
        assert incidences[index] == 16
    return rich


def verify_group(points: Sequence[Vector]) -> dict[str, object]:
    matrices = [quaternion_matrix(point) for point in points]
    matrix_index = {projective(matrix): i for i, matrix in enumerate(matrices)}
    assert len(matrix_index) == 60
    identity_key = projective(Q0)

    order_histogram: Counter[int] = Counter()
    for matrix in matrices:
        current = Q0
        for order in range(1, 7):
            current = matmul(current, matrix)
            if projective(current) == identity_key:
                order_histogram[order] += 1
                break
        else:
            raise AssertionError("group element has unexpected order")
        assert projective(mat_transpose(matrix)) in matrix_index
        assert projective(mat_inverse(matrix)) in matrix_index

    for left in matrices:
        for right in matrices:
            assert projective(matmul(left, right)) in matrix_index

    assert order_histogram == Counter({1: 1, 2: 15, 3: 20, 5: 24})
    return {
        "row_counts": [4, 8, 48],
        "order_histogram": {str(k): order_histogram[k]
                            for k in sorted(order_histogram)},
        "closed_under": ["multiplication", "inverse", "transpose"],
    }


def bits4(index: int) -> tuple[int, int, int, int]:
    return ((index >> 3) & 1, (index >> 2) & 1,
            (index >> 1) & 1, index & 1)


def mentry(matrix: Matrix2, row: int, column: int) -> E:
    return matrix[2 * row + column]


def product_tensor(matching: int, left: Matrix2, right: Matrix2) -> Vector:
    values = []
    for index in range(16):
        x1, x2, x3, x4 = bits4(index)
        if matching == 0:
            value = mentry(left, x1, x2) * mentry(right, x3, x4)
        elif matching == 1:
            value = mentry(left, x1, x3) * mentry(right, x2, x4)
        elif matching == 2:
            value = mentry(left, x1, x4) * mentry(right, x2, x3)
        else:
            raise ValueError("matching must be 0, 1, or 2")
        values.append(value)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class DeckRecord:
    matching: int
    left: int
    right: int
    vector: Vector


def generate_deck(points: Sequence[Vector]) -> tuple[list[DeckRecord], dict[Vector, int]]:
    matrices = [quaternion_matrix(point) for point in points]
    records: list[DeckRecord] = []
    index: dict[Vector, int] = {}
    for matching in range(3):
        for left in range(60):
            for right in range(60):
                vector = projective(product_tensor(
                    matching, matrices[left], matrices[right]))
                if vector in index:
                    raise AssertionError("duplicate projective deck point")
                index[vector] = len(records)
                records.append(DeckRecord(matching, left, right, vector))
    assert len(records) == len(index) == 10_800
    return records, index


def line_through_fixed_key(fixed: Vector, point: Vector) -> Vector:
    pivot = next(i for i, value in enumerate(fixed) if not value.is_zero())
    residual = vsub(point, vscale(point[pivot] / fixed[pivot], fixed))
    return projective(residual)


def lines_through_A(deck: Sequence[DeckRecord], a_index: int
                    ) -> tuple[dict[Vector, list[int]], Counter[int]]:
    fixed = deck[a_index].vector
    lines: dict[Vector, list[int]] = defaultdict(list)
    for index, record in enumerate(deck):
        if index == a_index:
            continue
        lines[line_through_fixed_key(fixed, record.vector)].append(index)
    histogram = Counter(len(indices) for indices in lines.values())
    assert histogram == Counter({1: 10_591, 2: 80, 4: 12})
    return lines, histogram


def solve_in_span(left: Vector, right: Vector, target: Vector) -> tuple[E, E]:
    assert len(left) == len(right) == len(target)
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            determinant = left[i] * right[j] - left[j] * right[i]
            if determinant.is_zero():
                continue
            a = (target[i] * right[j] - target[j] * right[i]) / determinant
            b = (left[i] * target[j] - left[j] * target[i]) / determinant
            assert vadd(vscale(a, left), vscale(b, right)) == target
            return a, b
    raise ValueError("vectors are projectively dependent")


def in_deck_cone(vector: Vector, deck_index: dict[Vector, int]) -> bool:
    return is_zero_vector(vector) or projective(vector) in deck_index


def r_vector(u: Sequence[E]) -> Vector:
    assert len(u) == 3
    a, b, d = u
    j = quaternion_matrix((a, b, ZERO, d))
    jbar = quaternion_matrix((a, -b, ZERO, -d))
    denominator = TWO * (a * a + b * b + d * d)
    return vscale(denominator.inverse(), product_tensor(2, j, jbar))


def bridge_formula(points: Sequence[Vector], a: Vector, b: Vector
                   ) -> tuple[list[Vector], list[dict[str, object]], dict[Vector, Vector]]:
    h_points = sorted((point for point in points if point[2].is_zero()),
                      key=vector_sort_key)
    assert len(h_points) == 15
    u_to_r: dict[Vector, Vector] = {}
    records: list[dict[str, object]] = [
        {"kind": "zero", "vector": vector_data(tuple(ZERO for _ in range(16)))},
        {"kind": "A/2", "vector": vector_data(vscale(Fraction(1, 2), a))},
        {"kind": "B/2", "vector": vector_data(vscale(Fraction(1, 2), b))},
    ]
    bridge = [tuple(ZERO for _ in range(16)),
              vscale(Fraction(1, 2), a),
              vscale(Fraction(1, 2), b)]
    for point in h_points:
        u = (point[0], point[1], point[3])
        value = r_vector(u)
        u_to_r[projective(u)] = value
        bridge.append(value)
        records.append({"kind": "R_u", "u": vector_data(u),
                        "vector": vector_data(value)})
    assert len(u_to_r) == 15
    assert len(bridge) == len(set(bridge)) == 18
    return bridge, records, u_to_r


def enumerate_bridge(deck: Sequence[DeckRecord], deck_index: dict[Vector, int],
                     a_index: int, b: Vector,
                     line_buckets: dict[Vector, list[int]]) -> set[Vector]:
    a = deck[a_index].vector
    candidates: set[Vector] = {
        tuple(ZERO for _ in range(16)),
        vscale(Fraction(1, 2), a),
    }

    # Nondependent branch: X and A-2X have distinct projective directions
    # on one of the 92 rich deck lines through A.
    for other_indices in line_buckets.values():
        if len(other_indices) < 2:
            continue
        full_line = [a_index] + other_indices
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

    # Dependent branch: X and A-2X both have direction A.  The possible
    # third directions are exactly the deck points on Span(A,B).
    b_line_key = line_through_fixed_key(a, projective(b))
    ab_members = [a_index] + line_buckets[b_line_key]
    assert len(ab_members) == 3
    for w_index in ab_members:
        if w_index == a_index:
            continue
        w = deck[w_index].vector
        coefficient_a, _ = solve_in_span(a, w, b)
        x = vscale(coefficient_a / TWO, a)
        if (in_deck_cone(x, deck_index)
                and in_deck_cone(vsub(a, vscale(TWO, x)), deck_index)
                and in_deck_cone(vsub(vscale(TWO, x), b), deck_index)):
            candidates.add(x)
    return candidates


def dot(left: Sequence[E], right: Sequence[E]) -> E:
    assert len(left) == len(right)
    return sum((a * b for a, b in zip(left, right)), ZERO)


def explicit_frames() -> list[list[Vector]]:
    raw = [
        [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
        [(1, -(TAU ** 2), TAU), (1, -PHI, -(PHI ** 2)),
         (1, PHI, -TAU)],
        [(1, -(TAU ** 2), -TAU), (1, -PHI, PHI ** 2),
         (1, PHI, TAU)],
        [(1, -PHI, TAU), (1, PHI, PHI ** 2),
         (1, TAU ** 2, -TAU)],
        [(1, -PHI, -TAU), (1, PHI, -(PHI ** 2)),
         (1, TAU ** 2, TAU)],
    ]
    return [[tuple(E.coerce(x) for x in u) for u in frame]
            for frame in raw]


def matrix_rank(rows: list[list[E]]) -> int:
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    work = [row[:] for row in rows]
    rank = 0
    for column in range(column_count):
        pivot = next((r for r in range(rank, row_count)
                      if not work[r][column].is_zero()), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column].inverse()
        work[rank] = [scale * entry for entry in work[rank]]
        for row in range(row_count):
            if row == rank or work[row][column].is_zero():
                continue
            factor = work[row][column]
            work[row] = [x - factor * y
                         for x, y in zip(work[row], work[rank])]
        rank += 1
        if rank == row_count:
            break
    return rank


def flattening_rank(vector: Vector,
                    row_positions: tuple[int, int]) -> int:
    column_positions = tuple(i for i in range(4) if i not in row_positions)
    matrix = [[ZERO for _ in range(4)] for _ in range(4)]
    for index, value in enumerate(vector):
        bits = bits4(index)
        row = 2 * bits[row_positions[0]] + bits[row_positions[1]]
        column = 2 * bits[column_positions[0]] + bits[column_positions[1]]
        matrix[row][column] = value
    return matrix_rank(matrix)


def contract_quaternary_pair(vector: Vector,
                             pair: tuple[int, int],
                             kernel: Matrix2) -> Matrix2:
    """Contract an ordered pair of q4 ports by the bilinear kernel pairing.

    Ports are numbered 0,...,3 internally.  The two surviving ports retain
    their increasing order, and the returned binary is in row-major order.
    Thus for pair (0,1), B(x3,x4) is
    sum_{x1,x2} q(x1,x2,x3,x4) K(x1,x2), with no conjugation.
    """
    assert len(vector) == 16
    assert 0 <= pair[0] < pair[1] < 4
    remaining = tuple(position for position in range(4)
                      if position not in pair)
    result: list[E] = []
    for surviving_bits in itertools.product((0, 1), repeat=2):
        value = ZERO
        for contracted_bits in itertools.product((0, 1), repeat=2):
            bits = [0, 0, 0, 0]
            bits[pair[0]], bits[pair[1]] = contracted_bits
            bits[remaining[0]], bits[remaining[1]] = surviving_bits
            tensor_index = (8 * bits[0] + 4 * bits[1]
                            + 2 * bits[2] + bits[3])
            value += (vector[tensor_index]
                      * mentry(kernel, contracted_bits[0], contracted_bits[1]))
        result.append(value)
    return tuple(result)  # type: ignore[return-value]


def matrix_determinant(matrix: Matrix2) -> E:
    return matrix[0] * matrix[3] - matrix[1] * matrix[2]


def jsonl_payload(kind: str, records: Sequence[dict[str, object]],
                  schema: str = SCHEMA) -> bytes:
    header = {
        "field": "(a+b*phi)+i*(c+d*phi), phi^2=phi+1",
        "object": kind,
        "records": len(records),
        "schema": schema,
    }
    all_records = [header, *records]
    text = "\n".join(json.dumps(record, sort_keys=True,
                                separators=(",", ":"), ensure_ascii=True)
                     for record in all_records)
    return text.encode("utf-8")


def build_certificate() -> tuple[dict[str, bytes], dict[str, object]]:
    points = generate_group_points()
    group_summary = verify_group(points)
    group_index = {point: index for index, point in enumerate(points)}
    q0_index = group_index[projective((ONE, ZERO, ZERO, ZERO))]
    q2_index = group_index[projective((ZERO, ZERO, ONE, ZERO))]

    domain_lines = domain_rich_lines(points)
    deck, deck_index = generate_deck(points)

    a = product_tensor(0, Q0, Q0)
    b = product_tensor(1, Q0, Q0)
    c = product_tensor(2, Q2, Q2)
    assert c == vsub(a, b)
    assert projective(a) == a and projective(b) == b
    a_index = deck_index[projective(a)]
    b_index = deck_index[projective(b)]
    c_index = deck_index[projective(c)]
    assert deck[a_index] == DeckRecord(0, q0_index, q0_index, a)
    assert deck[b_index] == DeckRecord(1, q0_index, q0_index, b)
    assert deck[c_index] == DeckRecord(2, q2_index, q2_index, c)

    line_buckets, line_histogram = lines_through_A(deck, a_index)
    rich_lines = sorted(tuple(sorted([a_index, *indices]))
                        for indices in line_buckets.values()
                        if len(indices) > 1)
    assert Counter(len(line) for line in rich_lines) == Counter({3: 80, 5: 12})
    assert len(rich_lines) == 92

    bridge, bridge_records, u_to_r = bridge_formula(points, a, b)
    enumerated_bridge = enumerate_bridge(
        deck, deck_index, a_index, b, line_buckets)
    assert enumerated_bridge == set(bridge)
    for x in bridge:
        assert in_deck_cone(x, deck_index)
        assert in_deck_cone(vsub(a, vscale(TWO, x)), deck_index)
        assert in_deck_cone(vsub(vscale(TWO, x), b), deck_index)

    bridge_index = {vector: index for index, vector in enumerate(bridge)}
    target = vscale(Fraction(1, 2), vadd(a, b))
    ordered_solutions = []
    for indices in itertools.product(range(18), repeat=3):
        if vsum((bridge[index] for index in indices), 16) == target:
            ordered_solutions.append(indices)
    assert len(ordered_solutions) == 36
    rank_two_solutions = [solution for solution in ordered_solutions
                          if set(solution) == {0, 1, 2}]
    assert len(rank_two_solutions) == 6
    nontrivial_solutions = [solution for solution in ordered_solutions
                            if solution not in rank_two_solutions]
    solution_frames = Counter(tuple(sorted(solution))
                              for solution in nontrivial_solutions)
    assert len(solution_frames) == 5
    assert set(solution_frames.values()) == {6}

    frame_us = explicit_frames()
    frame_indices: list[tuple[int, int, int]] = []
    for frame in frame_us:
        for i, j in itertools.combinations(range(3), 2):
            assert dot(frame[i], frame[j]).is_zero()
        indices = tuple(bridge_index[u_to_r[projective(u)]] for u in frame)
        assert len(set(indices)) == 3
        assert vsum((bridge[index] for index in indices), 16) == target
        frame_indices.append(indices)
    assert {tuple(sorted(frame)) for frame in frame_indices} == set(solution_frames)

    output_vectors: list[Vector] = []
    outputs = []
    support_histogram: Counter[int] = Counter()
    rank_histogram: Counter[tuple[int, int, int]] = Counter()
    for frame_number, indices in enumerate(frame_indices):
        for ordering in itertools.permutations(indices):
            output = vadd(
                vsub(vscale(PHI, bridge[ordering[0]]),
                     vscale(TAU, bridge[ordering[1]])),
                bridge[ordering[2]],
            )
            ranks = (
                flattening_rank(output, (0, 1)),
                flattening_rank(output, (0, 2)),
                flattening_rank(output, (0, 3)),
            )
            support = sum(not value.is_zero() for value in output)
            assert ranks == (3, 3, 3)
            output_vectors.append(output)
            outputs.append({
                "frame": frame_number,
                "ordering": list(ordering),
                "ranks": list(ranks),
                "support": support,
                "vector": vector_data(output),
            })
            rank_histogram[ranks] += 1
            support_histogram[support] += 1
    assert len(outputs) == 30
    assert rank_histogram == Counter({(3, 3, 3): 30})
    assert support_histogram == Counter({8: 6, 12: 4, 16: 20})

    # Terminal q4 exit.  The group point at index 24 is the fixed actual
    # kernel K_I = Q0 - tau^2 Q1 + tau Q3.  Contracting ports (1,2) in the
    # mathematical convention (internal positions (0,1)) gives a binary B;
    # group membership is tested on its transfer matrix B X, never on B.
    group_matrices = [quaternion_matrix(point) for point in points]
    group_transfer_keys = {projective(matrix) for matrix in group_matrices}
    uniform_kernel_coordinates = projective(
        (ONE, -(TAU ** 2), ZERO, TAU))
    uniform_kernel_index = group_index[uniform_kernel_coordinates]
    assert uniform_kernel_index == 24
    uniform_kernel = group_matrices[uniform_kernel_index]

    uniform_transfer_keys: set[Vector] = set()
    all_card_counts: Counter[str] = Counter()
    port_pairs = tuple(itertools.combinations(range(4), 2))
    for output, record in zip(output_vectors, outputs):
        binary = contract_quaternary_pair(output, (0, 1), uniform_kernel)
        determinant = matrix_determinant(binary)
        assert not determinant.is_zero()
        transfer_key = projective(matmul(binary, NATIVE_X))
        assert transfer_key not in group_transfer_keys
        uniform_transfer_keys.add(transfer_key)

        output_card_counts: Counter[str] = Counter()
        for pair in port_pairs:
            for kernel in group_matrices:
                card = contract_quaternary_pair(output, pair, kernel)
                if is_zero_vector(card):
                    classification = "zero"
                elif matrix_determinant(card).is_zero():
                    classification = "singular"
                elif projective(matmul(card, NATIVE_X)) in group_transfer_keys:
                    classification = "inside"
                else:
                    classification = "outside"
                output_card_counts[classification] += 1
                all_card_counts[classification] += 1

        record["uniform_outsider"] = {
            "binary": vector_data(binary),
            "determinant": determinant.to_data(),
            "kernel_group_index": uniform_kernel_index,
            "ports": [1, 2],
            "projective_transfer_BX": vector_data(transfer_key),
        }
        record["actual_card_counts"] = {
            kind: output_card_counts[kind]
            for kind in ("zero", "inside", "outside", "singular")
        }

    assert len(uniform_transfer_keys) == len(output_vectors) == 30
    assert all_card_counts == Counter({
        "zero": 180,
        "inside": 1260,
        "outside": 9360,
        "singular": 0,
    })

    group_records = [
        {"coordinates": vector_data(point), "index": index}
        for index, point in enumerate(points)
    ]
    domain_records = [
        {"points": list(line), "size": len(line)} for line in domain_lines
    ]
    deck_records = [
        {"index": index, "left": record.left, "matching": record.matching,
         "right": record.right, "vector": vector_data(record.vector)}
        for index, record in enumerate(deck)
    ]
    rich_records = [
        {"points": list(line), "size": len(line)} for line in rich_lines
    ]
    triple_records = [
        {"bridge_indices": list(solution),
         "kind": "rank_two" if set(solution) == {0, 1, 2} else "frame"}
        for solution in ordered_solutions
    ]
    frame_records = []
    for frame_number, (us, indices) in enumerate(zip(frame_us, frame_indices)):
        frame_records.append({
            "bridge_indices": list(indices),
            "frame": frame_number,
            "u": [vector_data(u) for u in us],
        })

    payloads = {
        "group_points.jsonl": jsonl_payload("group_points", group_records),
        "domain_lines.jsonl": jsonl_payload("domain_lines", domain_records),
        "deck_points.jsonl": jsonl_payload("deck_points", deck_records),
        "rich_lines_through_A.jsonl": jsonl_payload(
            "rich_lines_through_A", rich_records),
        "bridge_vectors.jsonl": jsonl_payload("bridge_vectors", bridge_records),
        "ordered_triples.jsonl": jsonl_payload(
            "ordered_triples", triple_records),
        "frames.jsonl": jsonl_payload("frames", frame_records),
        "kernel_outputs.jsonl": (
            jsonl_payload("kernel_outputs", outputs, KERNEL_OUTPUT_SCHEMA)
            + b"\n"
        ),
    }
    assert tuple(payloads) == PAYLOAD_NAMES

    summary: dict[str, object] = {
        "schema": SCHEMA,
        "group": group_summary,
        "domain_lines": {"3": 200, "5": 72},
        "deck_points": len(deck),
        "line_buckets_through_A": {
            str(size): line_histogram[size] for size in sorted(line_histogram)
        },
        "rich_lines_through_A": len(rich_lines),
        "bridge_vectors": len(bridge),
        "ordered_solutions": len(ordered_solutions),
        "rank_two_solutions": len(rank_two_solutions),
        "frames": len(frame_indices),
        "kernel_outputs": len(outputs),
        "flattening_rank_histogram": {"3,3,3": 30},
        "support_histogram": {
            str(size): support_histogram[size] for size in sorted(support_histogram)
        },
        "uniform_q4_outsider": {
            "distinct_projective_transfers": len(uniform_transfer_keys),
            "kernel_coordinates": vector_data(uniform_kernel_coordinates),
            "kernel_group_index": uniform_kernel_index,
            "outside_group": len(output_vectors),
            "ports": [1, 2],
            "transfer_convention": "B X",
        },
        "q4_actual_card_enumeration": {
            "cards": len(output_vectors) * len(port_pairs) * len(group_matrices),
            "inside": all_card_counts["inside"],
            "kernels": len(group_matrices),
            "outside": all_card_counts["outside"],
            "outputs": len(output_vectors),
            "port_pairs": len(port_pairs),
            "singular": all_card_counts["singular"],
            "zero": all_card_counts["zero"],
        },
    }
    return payloads, summary


def digest_manifest(payloads: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "payloads": {
            name: {
                "bytes": len(payloads[name]),
                "records": len(payloads[name].splitlines()) - 1,
                "schema": (KERNEL_OUTPUT_SCHEMA
                           if name == "kernel_outputs.jsonl" else SCHEMA),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
            for name in PAYLOAD_NAMES
        },
    }


def verify_manifest(computed: dict[str, object], path: Path) -> None:
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert expected == computed, (
        f"manifest mismatch: regenerate only after auditing schema changes: {path}")


def verify_payloads(computed: dict[str, bytes], directory: Path) -> None:
    for name in PAYLOAD_NAMES:
        path = directory / name
        assert path.is_file(), f"payload file not found: {path}"
        assert path.read_bytes() == computed[name], f"payload mismatch: {path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild and verify the bounded A5 certificate exactly."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest-v1.json"),
        help="manifest to compare with the regenerated data (default: manifest-v1.json)",
    )
    parser.add_argument(
        "--no-manifest-check",
        action="store_true",
        help=("skip manifest comparison and print JSON; payload comparison still "
              "runs unless --no-payload-check is also given"),
    )
    parser.add_argument(
        "--payload-directory",
        type=Path,
        default=Path(__file__).with_name("payloads-v1"),
        metavar="DIRECTORY",
        help="payload directory to compare with regenerated streams (default: payloads-v1)",
    )
    parser.add_argument(
        "--no-payload-check",
        action="store_true",
        help="skip byte-for-byte comparison with the checked-in payload streams",
    )
    parser.add_argument(
        "--write-payloads",
        type=Path,
        metavar="DIRECTORY",
        help="write the eight regenerated JSONL streams to DIRECTORY",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the verification result as JSON instead of the summary",
    )
    args = parser.parse_args()

    payloads, summary = build_certificate()
    manifest = digest_manifest(payloads)
    if not args.no_manifest_check:
        verify_manifest(manifest, args.manifest)
    if not args.no_payload_check:
        verify_payloads(payloads, args.payload_directory)

    if args.write_payloads is not None:
        args.write_payloads.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (args.write_payloads / name).write_bytes(payload)

    result = {
        "manifest": manifest,
        "manifest_checked": not args.no_manifest_check,
        "payloads_checked": not args.no_payload_check,
        "summary": summary,
    }
    if args.json or args.no_manifest_check:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.write_payloads is not None:
            print(f"payloads written to {args.write_payloads}")
        print("A5 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
