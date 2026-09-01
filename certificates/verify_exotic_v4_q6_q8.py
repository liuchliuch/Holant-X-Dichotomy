#!/usr/bin/env python3
"""Exact reconstruction certificate for the exotic-V4 q6/q8 strata.

The four binary directions are rebuilt from their displayed matrices over
Q(i).  No survivor table, checksum, random sample, floating-point arithmetic,
or finite-field specialization is used.  The q6 calculation fixes one nonzero
dual card by the physical local action and then scans the complete
``(48+1)^3`` card-state space.  Scale parameters are kept as exact linear
subspaces over Q(i), so the scan covers arbitrary complex coefficients rather
than a finite scale grid.

The q8 calculation is likewise reconstructed from matching data and exact
card maps.  Its stage counters document the finite state space being
exhausted; see ``main`` and the README for the precise certificate boundary.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Iterable, Sequence


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


@dataclass(frozen=True, slots=True)
class GaussianQ:
    """An exact element ``real + imag*i`` of Q(i)."""

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
Space = tuple[Vector, ...]
Matrix2 = tuple[Scalar, Scalar, Scalar, Scalar]


def q(value: GaussianQ | Fraction | int) -> GaussianQ:
    return GaussianQ.coerce(value)


def m2(values: Sequence[GaussianQ | Fraction | int]) -> Matrix2:
    assert len(values) == 4
    return tuple(q(value) for value in values)  # type: ignore[return-value]


E = (
    m2((1, 0, 0, 1)),
    m2((0, 1, 1, 0)),
    m2((I, 1, -1, -I)),
    m2((-I, 1, -1, I)),
)
DUAL_LABEL = (0, 1, 3, 2)
DUAL_SCALE = (q(Fraction(1, 2)), q(Fraction(1, 2)),
              q(Fraction(1, 4)), q(Fraction(1, 4)))


def entry(matrix: Matrix2, row: int, column: int) -> Scalar:
    return matrix[2 * row + column]


def frobenius(left: Matrix2, right: Matrix2) -> Scalar:
    return sum((left[index] * right[index] for index in range(4)), ZERO)


def matrix2_multiply(left: Matrix2, right: Matrix2) -> Matrix2:
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def projective(vector: Sequence[Scalar]) -> Vector:
    pivot = next(value for value in vector if value)
    return tuple(value / pivot for value in vector)


def projective_label(vector: Sequence[Scalar], lines: Sequence[Vector]) -> int:
    normalized = projective(vector)
    for label, line in enumerate(lines):
        if normalized == line:
            return label
    raise AssertionError("vector is outside the asserted projective line set")


def verify_binary_calculus() -> None:
    gram = tuple(tuple(frobenius(left, right) for right in E) for left in E)
    assert gram == (
        (q(2), ZERO, ZERO, ZERO),
        (ZERO, q(2), ZERO, ZERO),
        (ZERO, ZERO, ZERO, q(4)),
        (ZERO, ZERO, q(4), ZERO),
    )
    duals = tuple(tuple(DUAL_SCALE[label] * value
                        for value in E[DUAL_LABEL[label]])
                  for label in range(4))
    assert all(frobenius(duals[left], E[right])
               == (ONE if left == right else ZERO)
               for left in range(4) for right in range(4))
    normalized_binary_lines = tuple(projective(matrix) for matrix in E)
    assert {
        projective_label(matrix2_multiply(left, right), normalized_binary_lines)
        for left in E for right in E
    } == set(range(4))


def rref(rows: Iterable[Sequence[Scalar]], width: int) -> Space:
    work = [list(row) for row in rows]
    assert all(len(row) == width for row in work)
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, len(work))
                      if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [left - scale * right
                         for left, right in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return tuple(tuple(row) for row in work[:pivot_row])


def nullspace(rows: Iterable[Sequence[Scalar]], width: int) -> Space:
    reduced = rref(rows, width)
    pivots = tuple(next(column for column, value in enumerate(row) if value)
                   for row in reduced)
    output = []
    for free in range(width):
        if free in pivots:
            continue
        vector = [ZERO] * width
        vector[free] = ONE
        for row, pivot in reversed(tuple(zip(reduced, pivots))):
            vector[pivot] = -sum(
                (row[column] * vector[column] for column in range(width)),
                ZERO,
            )
        output.append(tuple(vector))
    return tuple(output)


def has_all_live_vector(space: Space, width: int) -> bool:
    # Q(i) is infinite: a finite union of coordinate hyperplanes cannot cover
    # a space unless one coordinate vanishes identically on that space.
    return bool(space) and all(any(row[column] for row in space)
                               for column in range(width))


def contains(large: Space, small: Space, width: int) -> bool:
    return len(rref((*large, *small), width)) == len(large)


def maximal_spaces(spaces: Iterable[Space], width: int) -> set[Space]:
    unique = set(spaces)
    return {
        left for left in unique
        if not any(left != right and len(left) <= len(right)
                   and contains(right, left, width) for right in unique)
    }


def intersection(left: Space, right: Space, width: int) -> Space:
    equations = [
        tuple([row[column] for row in left]
              + [-row[column] for row in right])
        for column in range(width)
    ]
    relations = nullspace(equations, len(left) + len(right))
    return rref(
        (
            tuple(sum((relation[index] * left[index][column]
                       for index in range(len(left))), ZERO)
                  for column in range(width))
            for relation in relations
        ),
        width,
    )


def bits(index: int, arity: int) -> tuple[int, ...]:
    return tuple((index >> (arity - 1 - port)) & 1 for port in range(arity))


def matching_products(arity: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    def rec(ports: tuple[int, ...]
            ) -> Iterable[tuple[tuple[int, int], ...]]:
        if not ports:
            yield ()
            return
        first = ports[0]
        for offset in range(1, len(ports)):
            second = ports[offset]
            remainder = ports[1:offset] + ports[offset + 1:]
            for rest in rec(remainder):
                yield ((first, second), *rest)

    return tuple(rec(tuple(range(arity))))


MATCHINGS4 = matching_products(4)
MATCHINGS6 = matching_products(6)
MATCHINGS8 = matching_products(8)
assert tuple(map(len, (MATCHINGS4, MATCHINGS6, MATCHINGS8))) == (3, 15, 105)


def product_tensor(arity: int, matching: Sequence[tuple[int, int]],
                   labels: Sequence[int]) -> Vector:
    output = []
    for word in range(1 << arity):
        word_bits = bits(word, arity)
        value = ONE
        for pair, label in zip(matching, labels):
            value *= entry(E[label], word_bits[pair[0]], word_bits[pair[1]])
        output.append(value)
    return tuple(output)


LINES4_RAW = tuple(
    product_tensor(4, matching, labels)
    for matching in MATCHINGS4
    for labels in itertools.product(range(4), repeat=2)
)
LINES4 = tuple(projective(line) for line in LINES4_RAW)
assert len(LINES4) == len(set(LINES4)) == 48
PROJECTIVE_LINES4 = set(LINES4)

_support_lines: dict[tuple[int, ...], list[Vector]] = defaultdict(list)
for _line in LINES4:
    _support_lines[tuple(index for index, value in enumerate(_line) if value)] \
        .append(_line)
SUPPORT_LINES4 = {support: tuple(lines)
                  for support, lines in _support_lines.items()}
assert len(SUPPORT_LINES4) == 25
assert Counter(map(len, SUPPORT_LINES4.values())) == Counter({1: 12, 2: 12, 12: 1})


def contract(tensor: Sequence[Scalar], arity: int, pair: tuple[int, int],
             kernel: Matrix2) -> Vector:
    remaining = tuple(port for port in range(arity) if port not in pair)
    output = []
    for residual in itertools.product((0, 1), repeat=arity - 2):
        total = ZERO
        for internal in itertools.product((0, 1), repeat=2):
            word_bits = [0] * arity
            word_bits[pair[0]], word_bits[pair[1]] = internal
            for port, value in zip(remaining, residual):
                word_bits[port] = value
            index = sum(value << (arity - 1 - port)
                        for port, value in enumerate(word_bits))
            total += tensor[index] * entry(kernel, *internal)
        output.append(total)
    return tuple(output)


def basis_tensor6(label: int, line: int) -> Vector:
    output = []
    residual = LINES4_RAW[line]
    for word in range(64):
        word_bits = bits(word, 6)
        output.append(entry(E[label], word_bits[0], word_bits[1])
                      * residual[word & 15])
    return tuple(output)


BASIS6 = tuple(tuple(basis_tensor6(label, line) for line in range(48))
               for label in range(4))
Q6_CONTEXTS = tuple(
    (pair, kernel_label)
    for pair in itertools.combinations(range(6), 2)
    if pair != (0, 1)
    for kernel_label in range(4)
)
assert len(Q6_CONTEXTS) == 56


def build_q6_contributions(
) -> dict[tuple[tuple[int, int], int], tuple[tuple[Vector, ...], ...]]:
    return {
        context: tuple(
            tuple(contract(BASIS6[label][line], 6, context[0], E[context[1]])
                  for line in range(48))
            for label in range(4)
        )
        for context in Q6_CONTEXTS
    }


def q6_columns(
    contributions: dict[tuple[tuple[int, int], int],
                        tuple[tuple[Vector, ...], ...]],
    context: tuple[tuple[int, int], int],
    assignment: tuple[int, int, int],
) -> tuple[int, tuple[Vector, ...]]:
    line_by_label = (0, *assignment)
    live = tuple(label for label, line in enumerate(line_by_label) if line >= 0)
    table = contributions[context]
    return len(live), tuple(table[label][line_by_label[label]] for label in live)


def allowed_line_preimages(columns: Sequence[Sequence[Scalar]],
                           support_lines: dict[tuple[int, ...], tuple[Vector, ...]],
                           output_width: int) -> set[Space]:
    variable_count = len(columns)
    candidates: set[Space] = set()
    for support, allowed_lines in support_lines.items():
        outside = tuple(index for index in range(output_width)
                        if index not in support)
        support_kernel = nullspace(
            (tuple(columns[column][row] for column in range(variable_count))
             for row in outside),
            variable_count,
        )
        if not has_all_live_vector(support_kernel, variable_count):
            continue
        if len(support_kernel) == 1:
            scale = support_kernel[0]
            output = tuple(
                sum((scale[column] * columns[column][row]
                     for column in range(variable_count)), ZERO)
                for row in range(output_width)
            )
            if not any(output) or projective(output) in set(allowed_lines):
                candidates.add(rref(support_kernel, variable_count))
            continue
        for allowed in allowed_lines:
            equations = tuple(
                tuple(columns[column][row]
                      for column in range(variable_count))
                + (-allowed[row],)
                for row in range(output_width)
            )
            lifted = nullspace(equations, variable_count + 1)
            projected = rref((row[:variable_count] for row in lifted),
                             variable_count)
            if has_all_live_vector(projected, variable_count):
                candidates.add(projected)
    return maximal_spaces(candidates, variable_count)


def q6_output(columns: Sequence[Sequence[Scalar]], scale: Sequence[Scalar]
              ) -> Vector:
    return tuple(sum((scale[column] * columns[column][row]
                      for column in range(len(columns))), ZERO)
                 for row in range(16))


def solve_q6_assignment(
    contributions: dict[tuple[tuple[int, int], int],
                        tuple[tuple[Vector, ...], ...]],
    assignment: tuple[int, int, int],
) -> set[Vector]:
    variable_count = 1 + sum(line >= 0 for line in assignment)
    order = sorted(Q6_CONTEXTS,
                   key=lambda context: (context[0][0] >= 2,
                                        context[0], context[1]))
    _, columns = q6_columns(contributions, order[0], assignment)
    spaces = allowed_line_preimages(columns, SUPPORT_LINES4, 16)
    for context in order[1:]:
        if not spaces or all(len(space) == 1 for space in spaces):
            break
        _, next_columns = q6_columns(contributions, context, assignment)
        next_spaces = allowed_line_preimages(next_columns, SUPPORT_LINES4, 16)
        common_spaces = []
        for left in spaces:
            for right in next_spaces:
                common = intersection(left, right, variable_count)
                if has_all_live_vector(common, variable_count):
                    common_spaces.append(common)
        spaces = maximal_spaces(common_spaces, variable_count)
    assert all(len(space) == 1 for space in spaces), (
        "continuous q6 scale family survived every exotic card", assignment,
        spaces,
    )
    solutions = set()
    for space in spaces:
        scale = projective(space[0])
        safe = True
        for context in Q6_CONTEXTS:
            _, context_columns = q6_columns(contributions, context, assignment)
            output = q6_output(context_columns, scale)
            if any(output) and projective(output) not in PROJECTIVE_LINES4:
                safe = False
                break
        if safe:
            solutions.add(scale)
    return solutions


def reconstruct_q6(assignment: tuple[int, int, int], scale: Sequence[Scalar]
                   ) -> Vector:
    lines = (0, *assignment)
    scale_by_label = [ZERO] * 4
    iterator = iter(scale)
    for label, line in enumerate(lines):
        if line >= 0:
            scale_by_label[label] = next(iterator)
    return tuple(
        sum((scale_by_label[label] * BASIS6[label][lines[label]][word]
             for label in range(4) if lines[label] >= 0), ZERO)
        for word in range(64)
    )


LINES6_RAW = tuple(
    product_tensor(6, matching, labels)
    for matching in MATCHINGS6
    for labels in itertools.product(range(4), repeat=3)
)
LINES6 = tuple(projective(line) for line in LINES6_RAW)
PRODUCTS6 = set(LINES6)
assert len(LINES6) == len(PRODUCTS6) == 960


def local_action4(tensor: Sequence[Scalar], port: int,
                  matrix: Matrix2) -> Vector:
    output = []
    for target in range(16):
        target_bits = bits(target, 4)
        value = ZERO
        for source_bit in (0, 1):
            source_bits = list(target_bits)
            source_bits[port] = source_bit
            source = sum(source_bits[index] << (3 - index)
                         for index in range(4))
            value += entry(matrix, target_bits[port], source_bit) * tensor[source]
        output.append(value)
    return tuple(output)


def permute4(tensor: Sequence[Scalar], source_at_target: Sequence[int]) -> Vector:
    assert sorted(source_at_target) == list(range(4))
    output = []
    for target in range(16):
        target_bits = bits(target, 4)
        source_bits = tuple(target_bits[source_at_target[port]]
                            for port in range(4))
        source = sum(source_bits[index] << (3 - index) for index in range(4))
        output.append(tensor[source])
    return tuple(output)


def residual_stabilizer() -> set[tuple[int, ...]]:
    """Rebuild the 64-element physical stabilizer of the fixed base card."""

    line_index = {line: index for index, line in enumerate(LINES4)}

    def induced(action) -> tuple[int, ...]:
        permutation = tuple(
            line_index[projective(action(line))] for line in LINES4_RAW
        )
        assert sorted(permutation) == list(range(48))
        assert permutation[0] == 0
        return permutation

    generators: list[tuple[int, ...]] = []
    base = LINES4_RAW[0]
    for left_port, right_port in ((0, 1), (2, 3)):
        for left_label in range(1, 4):
            right_label = next(
                label for label in range(4)
                if projective(local_action4(
                    local_action4(base, left_port, E[left_label]),
                    right_port,
                    E[label],
                )) == LINES4[0]
            )
            generators.append(induced(
                lambda tensor, lp=left_port, rp=right_port,
                ll=left_label, rl=right_label:
                local_action4(local_action4(tensor, lp, E[ll]), rp, E[rl])
            ))
    generators.extend((
        induced(lambda tensor: permute4(tensor, (1, 0, 2, 3))),
        induced(lambda tensor: permute4(tensor, (0, 1, 3, 2))),
        induced(lambda tensor: permute4(tensor, (2, 3, 0, 1))),
    ))

    def compose(left: tuple[int, ...], right: tuple[int, ...]
                ) -> tuple[int, ...]:
        return tuple(left[right[index]] for index in range(48))

    identity = tuple(range(48))
    group = {identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for generator in generators:
            product = compose(current, generator)
            if product not in group:
                group.add(product)
                queue.append(product)
    assert len(group) == 64
    return group


def q6_assignment_orbits(
    stabilizer: set[tuple[int, ...]],
) -> tuple[tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...]:
    seen: set[tuple[int, int, int]] = set()
    output = []
    for assignment in itertools.product(range(-1, 48), repeat=3):
        if assignment in seen:
            continue
        orbit = tuple(sorted({
            tuple(-1 if line < 0 else action[line] for line in assignment)
            for action in stabilizer
        }))
        seen.update(orbit)
        output.append((orbit[0], orbit))
    assert len(seen) == 49 ** 3
    assert len(output) == 4045
    assert Counter(len(orbit) for _, orbit in output) == Counter({
        1: 27,
        2: 49,
        4: 351,
        8: 521,
        16: 577,
        32: 1830,
        64: 690,
    })
    return tuple(output)


def scalar_against_raw_line(vector: Sequence[Scalar], raw_line: Sequence[Scalar]
                            ) -> Scalar:
    pivot = next(index for index, value in enumerate(raw_line) if value)
    scale = vector[pivot] / raw_line[pivot]
    assert all(vector[index] == scale * raw_line[index]
               for index in range(len(vector)))
    return scale


def normalized_q6_products() -> dict[tuple[int, int, int], set[Vector]]:
    """Enumerate, rather than store, every normalized product completion."""

    line_index = {line: index for index, line in enumerate(LINES4)}
    candidates: dict[tuple[int, int, int], set[Vector]] = defaultdict(set)
    duals = tuple(tuple(DUAL_SCALE[label] * value
                        for value in E[DUAL_LABEL[label]])
                  for label in range(4))
    for matching in MATCHINGS6:
        for labels in itertools.product(range(4), repeat=3):
            tensor = product_tensor(6, matching, labels)
            cards = tuple(contract(tensor, 6, (0, 1), dual)
                          for dual in duals)
            card_lines = tuple(
                -1 if not any(card) else line_index[projective(card)]
                for card in cards
            )
            if card_lines[0] != 0:
                continue
            live_scales = tuple(
                scalar_against_raw_line(card, LINES4_RAW[line])
                for card, line in zip(cards, card_lines) if line >= 0
            )
            candidates[card_lines[1:]].add(projective(live_scales))
    assert set(candidates) == {
        (-1, -1, -1),
        (1, 2, 3),
        (1, 3, 2),
        (4, 8, 12),
        (4, 12, 8),
    }
    assert Counter({assignment: len(scales)
                    for assignment, scales in candidates.items()}.values()) \
        == Counter({4: 4, 1: 1})
    q6_scale_patterns = {
        (ONE, ONE, q(Fraction(1, 2)), q(Fraction(1, 2))),
        (ONE, ONE, q(Fraction(-1, 2)), q(Fraction(-1, 2))),
        (ONE, q(-1), q(Fraction(1, 2)), q(Fraction(-1, 2))),
        (ONE, q(-1), q(Fraction(-1, 2)), q(Fraction(1, 2))),
    }
    assert all(candidates[assignment] == q6_scale_patterns
               for assignment in candidates if assignment != (-1, -1, -1))
    assert candidates[(-1, -1, -1)] == {(ONE,)}
    assert sum(map(len, candidates.values())) == 17
    return dict(candidates)


def verify_q6() -> tuple[Counter[int], int, int]:
    print("  q6: constructing 56 exact card maps ...", flush=True)
    contributions = build_q6_contributions()
    product_candidates = normalized_q6_products()
    stabilizer = residual_stabilizer()
    orbits = q6_assignment_orbits(stabilizer)
    candidate_assignments = set(product_candidates)
    for assignment in candidate_assignments:
        image_assignments = {
            tuple(-1 if line < 0 else action[line] for line in assignment)
            for action in stabilizer
        }
        assert image_assignments <= candidate_assignments

    by_live_cards: Counter[int] = Counter()
    for assignment, scales in product_candidates.items():
        by_live_cards[1 + sum(line >= 0 for line in assignment)] += len(scales)
    product_count = sum(by_live_cards.values())
    core_count = 0
    safe_orbit_count = 0
    covered_states = 0
    for number, (assignment, orbit) in enumerate(orbits, start=1):
        solutions = solve_q6_assignment(contributions, assignment)
        covered_states += len(orbit)
        if number % 500 == 0:
            print(f"    checked {number}/4045 stabilizer orbits "
                  f"({covered_states}/117649 card states)",
                  flush=True)
        expected = product_candidates.get(assignment, set())
        assert solutions == expected
        if not solutions:
            continue
        safe_orbit_count += 1
        for scale in solutions:
            tensor = reconstruct_q6(assignment, scale)
            if projective(tensor) not in PRODUCTS6:
                core_count += 1
    assert covered_states == 49 ** 3
    assert safe_orbit_count == 2
    assert by_live_cards == Counter({4: 16, 1: 1})
    assert product_count == 17
    assert core_count == 0
    return by_live_cards, product_count, core_count


Q6_PROFILE_CONTEXTS = tuple(
    (pair, kernel_label)
    for pair in itertools.combinations(range(6), 2)
    for kernel_label in range(4)
)
assert len(Q6_PROFILE_CONTEXTS) == 60


CardRecord = tuple[int, Scalar]


def card_record(card: Sequence[Scalar], line_index: dict[Vector, int]
                ) -> CardRecord:
    if not any(card):
        return -1, ZERO
    normalized = projective(card)
    assert normalized in line_index
    pivot = next(value for value in card if value)
    return line_index[normalized], pivot


def build_product6_profiles(
) -> tuple[tuple[tuple[CardRecord, ...], ...],
           dict[tuple[int, int, int, int], set[Vector]]]:
    """Build all 960 x 60 actual profiles and the dual base relation."""

    line_index = {line: index for index, line in enumerate(LINES4)}
    profiles = []
    for state in LINES6:
        profiles.append(tuple(
            card_record(contract(state, 6, pair, E[kernel]), line_index)
            for pair, kernel in Q6_PROFILE_CONTEXTS
        ))

    duals = tuple(tuple(DUAL_SCALE[label] * value
                        for value in E[DUAL_LABEL[label]])
                  for label in range(4))
    base_relation: dict[tuple[int, int, int, int], set[Vector]] = defaultdict(set)
    for state in LINES6:
        records = tuple(card_record(contract(state, 6, (0, 1), dual),
                                    line_index)
                        for dual in duals)
        directions = tuple(direction for direction, _ in records)
        scales = tuple(scale for _, scale in records)
        base_relation[directions].add(projective(scales))

    # A zero residual card is a legal q6 deck member and imposes no scale
    # condition.  It is not represented by any projective product line.
    base_relation[(-1, -1, -1, -1)].add((ZERO, ZERO, ZERO, ZERO))

    assert len(base_relation) == 385
    assert sum(map(len, base_relation.values())) == 961
    assert Counter(sum(direction >= 0 for direction in directions)
                   for directions in base_relation) \
        == Counter({4: 192, 1: 192, 0: 1})
    assert Counter(map(len, base_relation.values())) == Counter({1: 193, 4: 192})
    return tuple(profiles), dict(base_relation)


def direction_compatible_triples(
    profiles: tuple[tuple[CardRecord, ...], ...],
    base_relation: dict[tuple[int, int, int, int], set[Vector]],
) -> tuple[tuple[int, int, int], ...]:
    """Apply all 60 direction-only constraints to the 960^3 triples."""

    state_count = len(profiles)
    all_states = (1 << state_count) - 1
    masks_by_context = []
    third_directions = []
    for context in range(len(Q6_PROFILE_CONTEXTS)):
        masks: dict[int, int] = defaultdict(int)
        for state in range(state_count):
            masks[profiles[state][context][0]] |= 1 << state
        masks_by_context.append(dict(masks))

        first_direction = profiles[0][context][0]
        relation: dict[tuple[int, int], set[int]] = defaultdict(set)
        for directions in base_relation:
            if directions[0] == first_direction:
                relation[directions[1], directions[2]].add(directions[3])
        third_directions.append(dict(relation))

    candidates = []
    for first in range(state_count):
        for second in range(state_count):
            mask = all_states
            for context in range(len(Q6_PROFILE_CONTEXTS)):
                key = (profiles[first][context][0],
                       profiles[second][context][0])
                options = third_directions[context].get(key)
                if not options:
                    mask = 0
                    break
                allowed = 0
                for direction in options:
                    allowed |= masks_by_context[context].get(direction, 0)
                mask &= allowed
                if not mask:
                    break
            while mask:
                low_bit = mask & -mask
                mask -= low_bit
                candidates.append((first, second, low_bit.bit_length() - 1))
    return tuple(candidates)


def local_lambda_spaces(records: Sequence[CardRecord],
                        target_patterns: Iterable[Vector]) -> set[Space]:
    """Return the union of local four-coefficient solution subspaces."""

    spaces = set()
    for target in target_patterns:
        live = tuple(index for index, (_, scale) in enumerate(records) if scale)
        if any(bool(target[index]) != (index in live) for index in range(4)):
            continue
        linked = [ZERO] * 4
        for index in live:
            linked[index] = target[index] / records[index][1]
        basis = [tuple(linked)]
        for index in range(4):
            if index not in live:
                axis = [ZERO] * 4
                axis[index] = ONE
                basis.append(tuple(axis))
        space = rref(basis, 4)
        if has_all_live_vector(space, 4):
            spaces.add(space)
    return maximal_spaces(spaces, 4)


def allowed_q8_scales(
    states: tuple[int, int, int, int],
    profiles: tuple[tuple[CardRecord, ...], ...],
    base_relation: dict[tuple[int, int, int, int], set[Vector]],
) -> set[Vector]:
    spaces: set[Space] = {
        tuple(tuple(ONE if row == column else ZERO for column in range(4))
              for row in range(4))
    }
    for context in range(len(Q6_PROFILE_CONTEXTS)):
        records = tuple(profiles[state][context] for state in states)
        directions = tuple(direction for direction, _ in records)
        if directions not in base_relation:
            return set()
        local = local_lambda_spaces(records, base_relation[directions])
        next_spaces = []
        for current in spaces:
            for allowed in local:
                common = intersection(current, allowed, 4)
                if has_all_live_vector(common, 4):
                    next_spaces.append(common)
        spaces = maximal_spaces(next_spaces, 4)
        if not spaces:
            return set()
    assert all(len(space) == 1 for space in spaces), (
        "continuous q8 scale family survived all residual cards", states, spaces
    )
    return {projective(space[0]) for space in spaces}


def reconstruct_q8(states: Sequence[int], scale: Sequence[Scalar]) -> Vector:
    output = []
    for word in range(256):
        word_bits = bits(word, 8)
        residual = word & 63
        output.append(sum(
            (scale[label]
             * entry(E[label], word_bits[0], word_bits[1])
             * LINES6[states[label]][residual]
             for label in range(4)),
            ZERO,
        ))
    return tuple(output)


def q8_factor_cuts(tensor: Sequence[Scalar]) -> tuple[int, ...]:
    """Return canonical nontrivial bipartitions with flattening rank one."""

    cuts = []
    for mask in range(1, 1 << 8):
        complement = ((1 << 8) - 1) ^ mask
        if mask > complement:
            continue
        left_ports = tuple(port for port in range(8) if mask & (1 << port))
        right_ports = tuple(port for port in range(8) if not mask & (1 << port))
        rows = 1 << len(left_ports)
        columns = 1 << len(right_ports)
        table = [[ZERO] * columns for _ in range(rows)]
        for word in range(256):
            word_bits = bits(word, 8)
            left = sum(word_bits[port] << (len(left_ports) - 1 - index)
                       for index, port in enumerate(left_ports))
            right = sum(word_bits[port] << (len(right_ports) - 1 - index)
                        for index, port in enumerate(right_ports))
            table[left][right] = tensor[word]
        pivot = next(((row, column) for row in range(rows)
                      for column in range(columns) if table[row][column]), None)
        assert pivot is not None
        pivot_row, pivot_column = pivot
        pivot_value = table[pivot_row][pivot_column]
        if all(table[row][column] * pivot_value
               == table[row][pivot_column] * table[pivot_row][column]
               for row in range(rows) for column in range(columns)):
            cuts.append(mask)
    return tuple(cuts)


def q8_globally_safe(tensor: Sequence[Scalar]) -> bool:
    for pair in itertools.combinations(range(8), 2):
        for kernel in E:
            card = contract(tensor, 8, pair, kernel)
            if any(card) and projective(card) not in PRODUCTS6:
                return False
    return True


def q8_direction_presentation_atlas(
    exact_triples: Sequence[tuple[int, int, int]],
    scale_triples: Sequence[tuple[tuple[int, int, int, int], set[Vector]]],
) -> tuple[int, int, int]:
    """Rebuild the specified ``216 -> 12 -> 48`` presentation atlas.

    A fixed-ruling q8 normal form has three independently chosen nonzero
    direction maps in GL(2,2)=S_3.  Support compatibility says that, at each
    of the three nonzero labels, the three images are pairwise distinct.
    Simultaneously reversing the two unmarked labels in the first direction
    does not change the resulting projective tensor.  The exact profile scan
    above applies after this two-to-one quotient.
    """

    nonzero = (1, 2, 3)
    directions = tuple(itertools.permutations(nonzero))
    presentations = tuple(itertools.product(directions, repeat=3))
    assert len(directions) == 6
    assert len(presentations) == 216

    compatible = tuple(
        triple for triple in presentations
        if all(len({triple[index][label - 1] for index in range(3)}) == 3
               for label in nonzero)
    )
    assert len(compatible) == 12

    def inverse(permutation: tuple[int, int, int]) -> dict[int, int]:
        return {image: source for source, image in zip(nonzero, permutation)}

    def exact_state_triple(
        triple: tuple[tuple[int, int, int], ...],
    ) -> tuple[int, int, int]:
        first, second, _ = triple
        first_map = dict(zip(nonzero, first))
        second_map = dict(zip(nonzero, second))
        first_inverse = inverse(first)
        relative = {
            label: second_map[first_inverse[label]] for label in nonzero
        }
        assert {relative[label] for label in nonzero} == set(nonzero)
        assert all(relative[label] != label for label in nonzero)
        positive = relative[1] == 2
        coordinate = first_map[1]

        # q6 lines use j=64*m+16*a+4*b+c.  Here m=0 and exactly one
        # factor label is nonzero; ``coordinate`` chooses that factor.
        place_value = {1: 16, 2: 4, 3: 1}[coordinate]
        order = (1, 2, 3) if positive else (1, 3, 2)
        return tuple(place_value * label for label in order)

    presentation_to_exact = {
        presentation: exact_state_triple(presentation)
        for presentation in compatible
    }
    image_histogram = Counter(presentation_to_exact.values())
    assert set(image_histogram) == set(exact_triples)
    assert set(image_histogram.values()) == {2}

    exact_scale_map = {states[1:]: scales for states, scales in scale_triples}
    assert set(exact_scale_map) == set(exact_triples)
    common_scale_patterns = next(iter(exact_scale_map.values()))
    assert len(common_scale_patterns) == 4
    assert all(scales == common_scale_patterns
               for scales in exact_scale_map.values())
    expected_patterns = {
        (ONE, ONE, -I / q(2), I / q(2)),
        (ONE, ONE, I / q(2), -I / q(2)),
        (ONE, q(-1), -I / q(2), -I / q(2)),
        (ONE, q(-1), I / q(2), I / q(2)),
    }
    assert common_scale_patterns == expected_patterns

    oriented_completions = {
        (presentation, scale)
        for presentation in compatible
        for scale in common_scale_patterns
    }
    exact_completions = {
        (presentation_to_exact[presentation], scale)
        for presentation, scale in oriented_completions
    }
    assert len(oriented_completions) == 48
    assert len(exact_completions) == 24
    assert Counter((presentation_to_exact[presentation], scale)
                   for presentation, scale in oriented_completions) \
        == Counter({completion: 2 for completion in exact_completions})
    return len(presentations), len(compatible), len(oriented_completions)


def verify_q8() -> tuple[int, int, int, int]:
    """Reconstruct the all-live q8 stratum in four auditable stages."""

    print("  q8: constructing all 960 x 60 product-card profiles ...", flush=True)
    profiles, base_relation = build_product6_profiles()
    direction_candidates = direction_compatible_triples(profiles, base_relation)
    print(f"    direction-compatible triples: {len(direction_candidates)}",
          flush=True)
    assert len(direction_candidates) == 6

    scale_triples: list[tuple[tuple[int, int, int, int], set[Vector]]] = []
    for triple in direction_candidates:
        states = (0, *triple)
        scales = allowed_q8_scales(states, profiles, base_relation)
        if scales:
            scale_triples.append((states, scales))
    scale_line_count = sum(len(scales) for _, scales in scale_triples)
    print(f"    globally scale-consistent triples: {len(scale_triples)}; "
          f"projective scale lines: {scale_line_count}", flush=True)
    assert len(scale_triples) == 6
    assert scale_line_count == 24

    irreducible = 0
    globally_safe = 0
    cut_sizes: Counter[int] = Counter()
    survivor_lines: set[Vector] = set()
    for states, scales in scale_triples:
        for scale in scales:
            tensor = reconstruct_q8(states, scale)
            assert tensor and any(tensor)
            if q8_globally_safe(tensor):
                globally_safe += 1
            cuts = q8_factor_cuts(tensor)
            if not cuts:
                irreducible += 1
            else:
                cut_sizes.update(min(mask.bit_count(), 8 - mask.bit_count())
                                 for mask in cuts)
            survivor_lines.add(projective(tensor))

    print(f"    literal tensor lines: {len(survivor_lines)}; "
          f"globally safe: {globally_safe}; irreducible: {irreducible}",
          flush=True)
    print(f"    factor-cut size incidences: {dict(sorted(cut_sizes.items()))}",
          flush=True)
    assert len(survivor_lines) == globally_safe == 24
    assert irreducible == 0
    assert cut_sizes == Counter({2: 96, 4: 72})

    presentation_counts = q8_direction_presentation_atlas(
        direction_candidates, scale_triples
    )
    assert presentation_counts == (216, 12, 48)
    print("    oriented direction atlas: presentations=216, "
          "compatible triples=12, oriented completions=48; "
          "projective quotient: 6 directions -> 24 completions", flush=True)
    return (*presentation_counts, irreducible)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = perf_counter()
    print("Exotic-V4 q6/q8 exact certificate replay", flush=True)
    verify_binary_calculus()
    print("  binary basis, duals, and projective fusion: PASS", flush=True)
    q6_start = perf_counter()
    by_live, products, cores = verify_q6()
    print(
        "  q6 complete 49^3 state space: "
        f"safe scale lines by live-card count="
        f"{dict(sorted(by_live.items()))}; "
        f"product lines={products}; nonproduct core lines={cores} "
        f"({perf_counter() - q6_start:.3f}s)",
        flush=True,
    )
    q8_counts = verify_q8()
    print(
        "  q8 direction atlas: "
        f"presentations={q8_counts[0]}, "
        f"compatible presentations={q8_counts[1]}, "
        f"oriented completions={q8_counts[2]}; "
        f"irreducible survivors={q8_counts[3]}",
        flush=True,
    )
    print(
        "EXOTIC-V4 Q6/Q8 EXACT CERTIFICATE: PASS "
        f"({perf_counter() - start:.3f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
