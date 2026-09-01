#!/usr/bin/env python3
"""Exact exhaustive replay of the six-port full-V4/Pauli certificate.

The normalized first Bell card is fixed to B00 tensor B00.  Each of the
other three Bell cards is either zero or one of the 48 four-port Pauli
matching lines.  Thus the *complete* normalized card state space has

    (48 + 1)^3 = 117,649

members.  Scales are not discretized.  For every card state, this verifier
intersects exact rational preimage subspaces for actual Pauli cards.  Only
when the surviving scale space is one-dimensional does it normalize the
scale vector and check all 56 non-base pair/kernel contexts.  Consequently
the conclusion for arbitrary complex scales follows from rational linear
equations; the program does not merely try signs or a finite scale grid.

All tensor entries and all constraint matrices are rational.  Python's
``Fraction`` is used throughout; there is no floating point arithmetic,
random sampling, digest comparison, or finite-field specialization.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Iterable, Sequence


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


Q = Fraction
Vector = tuple[Q, ...]
Basis = tuple[Vector, ...]


def bell(label: int) -> tuple[int, int, int, int]:
    d, ell = divmod(label, 2)
    return tuple(((-1) ** (ell * x)) if (x ^ y) == d else 0
                 for x in (0, 1) for y in (0, 1))


BELLS = tuple(bell(label) for label in range(4))
MATCHINGS4 = (((0, 1), (2, 3)),
              ((0, 2), (1, 3)),
              ((0, 3), (1, 2)))


def product4(matching: int, left_label: int, right_label: int
             ) -> tuple[int, ...]:
    edges = MATCHINGS4[matching]
    left, right = BELLS[left_label], BELLS[right_label]
    output = []
    for bits in itertools.product((0, 1), repeat=4):
        output.append(
            left[2 * bits[edges[0][0]] + bits[edges[0][1]]]
            * right[2 * bits[edges[1][0]] + bits[edges[1][1]]]
        )
    return tuple(output)


LINES4 = tuple(product4(matching, left, right)
               for matching in range(3)
               for left in range(4) for right in range(4))
assert len(LINES4) == len(set(LINES4)) == 48

SUPPORT_LINES: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
_support_lines: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
for _line in LINES4:
    _support_lines[tuple(index for index, value in enumerate(_line) if value)] \
        .append(_line)
SUPPORT_LINES = {support: tuple(lines)
                 for support, lines in _support_lines.items()}
assert len(SUPPORT_LINES) == 12
assert set(map(len, SUPPORT_LINES.values())) == {4}


def q_projective(vector: Sequence[Q]) -> Vector:
    pivot = next(value for value in vector if value)
    return tuple(value / pivot for value in vector)


PROJECTIVE_LINES4 = {q_projective(tuple(map(Q, line))) for line in LINES4}


def rref_basis(rows: Iterable[Sequence[Q]], columns: int) -> Basis:
    work = [list(map(Q, row)) for row in rows]
    if not work:
        return ()
    pivot_row = 0
    for column in range(columns):
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
            factor = work[row][column]
            work[row] = [x - factor * y
                         for x, y in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return tuple(tuple(row) for row in work[:pivot_row])


def nullspace(rows: Iterable[Sequence[Q]], columns: int) -> Basis:
    reduced = rref_basis(rows, columns)
    pivots = []
    for row in reduced:
        pivots.append(next(index for index, value in enumerate(row) if value))
    free = [column for column in range(columns) if column not in pivots]
    basis = []
    for free_column in free:
        vector = [Q(0) for _ in range(columns)]
        vector[free_column] = Q(1)
        for row, pivot in reversed(list(zip(reduced, pivots))):
            vector[pivot] = -sum(row[column] * vector[column]
                                 for column in free)
        basis.append(tuple(vector))
    return tuple(basis)


def canonical_span(rows: Iterable[Sequence[Q]], columns: int) -> Basis:
    return rref_basis(rows, columns)


def has_all_live_vector(space: Basis, columns: int) -> bool:
    # Over the infinite field Q, a subspace contains a vector avoiding every
    # coordinate hyperplane iff no coordinate vanishes identically on it.
    return bool(space) and all(any(vector[column] for vector in space)
                               for column in range(columns))


def span_contains(large: Basis, small: Basis, columns: int) -> bool:
    return len(canonical_span((*large, *small), columns)) == len(large)


def maximal_spaces(spaces: Iterable[Basis], columns: int) -> set[Basis]:
    unique = set(spaces)
    return {space for space in unique
            if not any(space != other and len(space) <= len(other)
                       and span_contains(other, space, columns)
                       for other in unique)}


def intersection(left: Basis, right: Basis, columns: int) -> Basis:
    equations = []
    for coordinate in range(columns):
        equations.append(tuple(
            [vector[coordinate] for vector in left]
            + [-vector[coordinate] for vector in right]
        ))
    relations = nullspace(equations, len(left) + len(right))
    vectors = []
    for relation in relations:
        vector = tuple(sum(relation[index] * left[index][coordinate]
                           for index in range(len(left)))
                       for coordinate in range(columns))
        if any(vector):
            vectors.append(vector)
    return canonical_span(vectors, columns)


def six_bits(index: int) -> tuple[int, ...]:
    return tuple((index >> (5 - position)) & 1 for position in range(6))


def basis_tensor(card_label: int, line_index: int) -> tuple[int, ...]:
    base = BELLS[card_label]
    residual = LINES4[line_index]
    return tuple(
        base[2 * bits[0] + bits[1]]
        * residual[8 * bits[2] + 4 * bits[3] + 2 * bits[4] + bits[5]]
        for bits in map(six_bits, range(64))
    )


BASIS_TENSORS = tuple(tuple(basis_tensor(label, line)
                            for line in range(48))
                      for label in range(4))


Context = tuple[tuple[int, int], int]
CONTEXTS: tuple[Context, ...] = tuple(
    (pair, kernel) for pair in itertools.combinations(range(6), 2)
    if pair != (0, 1) for kernel in range(4)
)
assert len(CONTEXTS) == 56


def contract_basis(tensor: Sequence[int], pair: tuple[int, int],
                   kernel_label: int) -> tuple[int, ...]:
    remaining = tuple(port for port in range(6) if port not in pair)
    kernel = BELLS[kernel_label]
    output = []
    for residual in itertools.product((0, 1), repeat=4):
        total = 0
        for internal in itertools.product((0, 1), repeat=2):
            bits = [0] * 6
            bits[pair[0]], bits[pair[1]] = internal
            for port, value in zip(remaining, residual):
                bits[port] = value
            index = sum(value << (5 - port) for port, value in enumerate(bits))
            total += tensor[index] * kernel[2 * internal[0] + internal[1]]
        output.append(total)
    return tuple(output)


CONTRIBUTIONS: dict[Context, tuple[tuple[tuple[int, ...], ...], ...]] = {}


def initialize_contributions() -> None:
    """Build the exact card-map table after command-line parsing."""
    print("V4 q6 exact certificate replay")
    print("  constructing all rational card maps ...", flush=True)
    for context in CONTEXTS:
        pair, kernel = context
        CONTRIBUTIONS[context] = tuple(tuple(
            contract_basis(BASIS_TENSORS[label][line], pair, kernel)
            for line in range(48)
        ) for label in range(4))


def card_columns(context: Context, assignment: tuple[int, int, int]
                 ) -> tuple[list[int], tuple[tuple[int, ...], ...]]:
    """Return live Bell labels and their output columns.

    Assignment entries are -1 for a zero base card and 0,...,47 for a live
    matching line.  Bell label zero is fixed live on line zero.
    """
    line_by_label = (0, *assignment)
    live = [label for label, line in enumerate(line_by_label) if line >= 0]
    table = CONTRIBUTIONS[context]
    return live, tuple(table[label][line_by_label[label]] for label in live)


def allowed_scale_spaces(columns: Sequence[Sequence[int]]) -> set[Basis]:
    variable_count = len(columns)
    candidates: set[Basis] = set()
    for support, allowed_lines in SUPPORT_LINES.items():
        outside = [index for index in range(16) if index not in support]
        support_kernel = nullspace(
            (tuple(Q(columns[column][row]) for column in range(variable_count))
             for row in outside),
            variable_count,
        )
        if not has_all_live_vector(support_kernel, variable_count):
            continue
        if len(support_kernel) == 1:
            scale = support_kernel[0]
            output = tuple(sum(scale[column] * columns[column][row]
                               for column in range(variable_count))
                           for row in range(16))
            if (not any(output)
                    or q_projective(output) in PROJECTIVE_LINES4):
                candidates.add(canonical_span(support_kernel, variable_count))
            continue

        for allowed in allowed_lines:
            equations = [
                tuple(Q(columns[column][row])
                      for column in range(variable_count))
                + (Q(-allowed[row]),)
                for row in range(16)
            ]
            lifted = nullspace(equations, variable_count + 1)
            projected = canonical_span(
                (vector[:variable_count] for vector in lifted), variable_count
            )
            if has_all_live_vector(projected, variable_count):
                candidates.add(projected)
    return maximal_spaces(candidates, variable_count)


def output_for_scale(context: Context, assignment: tuple[int, int, int],
                     scale: Sequence[Q]) -> Vector:
    _, columns = card_columns(context, assignment)
    return tuple(sum(scale[column] * columns[column][row]
                     for column in range(len(columns)))
                 for row in range(16))


def scale_is_globally_safe(assignment: tuple[int, int, int],
                           scale: Sequence[Q]) -> bool:
    for context in CONTEXTS:
        output = output_for_scale(context, assignment, scale)
        if any(output) and q_projective(output) not in PROJECTIVE_LINES4:
            return False
    return True


def direct_card(tensor: Sequence[Q], pair: tuple[int, int],
                kernel_label: int) -> Vector:
    remaining = tuple(port for port in range(6) if port not in pair)
    kernel = BELLS[kernel_label]
    output = []
    for residual in itertools.product((0, 1), repeat=4):
        total = Q(0)
        for internal in itertools.product((0, 1), repeat=2):
            bits = [0] * 6
            bits[pair[0]], bits[pair[1]] = internal
            for port, value in zip(remaining, residual):
                bits[port] = value
            index = sum(value << (5 - port) for port, value in enumerate(bits))
            total += tensor[index] * kernel[2 * internal[0] + internal[1]]
        output.append(total)
    return tuple(output)


def solve_assignment(assignment: tuple[int, int, int]) -> set[Vector]:
    variable_count = 1 + sum(line >= 0 for line in assignment)
    # Cross-pair contexts are strongest and are used first.  The remaining
    # contexts are still checked exactly on every one-dimensional survivor.
    order = sorted(CONTEXTS,
                   key=lambda context: (context[0][0] >= 2,
                                        context[0], context[1]))
    live, columns = card_columns(order[0], assignment)
    spaces = allowed_scale_spaces(columns)
    if not spaces:
        return set()
    for context in order[1:]:
        if all(len(space) == 1 for space in spaces):
            break
        _, next_columns = card_columns(context, assignment)
        next_spaces = allowed_scale_spaces(next_columns)
        intersections = []
        for left in spaces:
            for right in next_spaces:
                common = intersection(left, right, variable_count)
                if has_all_live_vector(common, variable_count):
                    intersections.append(common)
        spaces = maximal_spaces(intersections, variable_count)
        if not spaces:
            return set()
    assert all(len(space) == 1 for space in spaces), (
        "continuous scale family survived all Pauli contexts", assignment, spaces
    )
    solutions = set()
    for space in spaces:
        scale = q_projective(space[0])
        if scale_is_globally_safe(assignment, scale):
            solutions.add(scale)
    return solutions


MATCHINGS6: list[tuple[tuple[int, int], ...]] = []


def enumerate_matchings(ports: tuple[int, ...]
                        ) -> Iterable[tuple[tuple[int, int], ...]]:
    if not ports:
        yield ()
        return
    first = ports[0]
    for index in range(1, len(ports)):
        second = ports[index]
        remainder = ports[1:index] + ports[index + 1:]
        for rest in enumerate_matchings(remainder):
            yield ((first, second), *rest)


MATCHINGS6 = list(enumerate_matchings(tuple(range(6))))
assert len(MATCHINGS6) == 15


def product6(matching: Sequence[tuple[int, int]], labels: Sequence[int]
             ) -> tuple[int, ...]:
    output = []
    for bits in map(six_bits, range(64)):
        value = 1
        for edge, label in zip(matching, labels):
            value *= BELLS[label][2 * bits[edge[0]] + bits[edge[1]]]
        output.append(value)
    return tuple(output)


PRODUCT6 = {
    q_projective(tuple(map(Q, product6(matching, labels))))
    for matching in MATCHINGS6
    for labels in itertools.product(range(4), repeat=3)
}
assert len(PRODUCT6) == 960


def reconstruct_tensor(assignment: tuple[int, int, int],
                       live_scale: Sequence[Q]) -> Vector:
    lines = (0, *assignment)
    scale_by_label = [Q(0)] * 4
    scale_iterator = iter(live_scale)
    for label, line in enumerate(lines):
        if line >= 0:
            scale_by_label[label] = next(scale_iterator)
    return tuple(sum(scale_by_label[label]
                     * BASIS_TENSORS[label][lines[label]][row]
                     for label in range(4) if lines[label] >= 0)
                 for row in range(64))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    initialize_contributions()
    safe: dict[tuple[int, int, int], set[Vector]] = {}
    by_live_cards = Counter()
    product_count = core_count = 0
    candidate_card_histogram = Counter()
    all_live_pairs: set[tuple[tuple[int, int, int], Vector]] = set()

    for number, assignment in enumerate(
        itertools.product(range(-1, 48), repeat=3), start=1
    ):
        solutions = solve_assignment(assignment)
        if number % 20_000 == 0:
            print(f"  scanned {number}/117649 normalized card states", flush=True)
        if not solutions:
            continue
        safe[assignment] = solutions
        live_count = 1 + sum(line >= 0 for line in assignment)
        by_live_cards[live_count] += len(solutions)
        for scale in solutions:
            tensor = reconstruct_tensor(assignment, scale)
            if q_projective(tensor) in PRODUCT6:
                product_count += 1
            else:
                core_count += 1
            if live_count == 4:
                all_live_pairs.add((assignment, scale))
                for pair in itertools.combinations(range(6), 2):
                    for kernel_label in range(4):
                        card = direct_card(tensor, pair, kernel_label)
                        if not any(card):
                            candidate_card_histogram["zero"] += 1
                        else:
                            assert q_projective(card) in PROJECTIVE_LINES4
                            candidate_card_histogram["matching"] += 1

    expected_triples = {
        (1, 2, 3), (4, 8, 12), (11, 13, 6), (14, 7, 9)
    }
    assert {assignment for assignment, _ in all_live_pairs} == expected_triples
    assert len(all_live_pairs) == 32
    assert candidate_card_histogram == Counter({"zero": 144,
                                                 "matching": 1776})
    for assignment, scale in all_live_pairs:
        assert set(scale).issubset({Q(1), Q(-1)})
        assert scale[0] == 1

    # The first two normalized triples are product tensors; the crossed two
    # are precisely the sixteen local-Pauli H6 completions.
    assert sum(assignment in {(1, 2, 3), (4, 8, 12)}
               for assignment, _ in all_live_pairs) == 16
    assert sum(assignment in {(11, 13, 6), (14, 7, 9)}
               for assignment, _ in all_live_pairs) == 16
    for assignment, scale in all_live_pairs:
        tensor = reconstruct_tensor(assignment, scale)
        is_product = q_projective(tensor) in PRODUCT6
        assert is_product == (assignment in {(1, 2, 3), (4, 8, 12)})

    canonical_h6 = reconstruct_tensor((14, 7, 9), (Q(1),) * 4)
    for index, value in enumerate(canonical_h6):
        x = six_bits(index)
        exponent = (x[0] + x[0] * x[1] + x[0] * x[5]
                    + x[1] * x[4] + x[2] * x[4] + x[2] * x[5])
        expected = Q((-1) ** exponent) if sum(x) % 2 == 0 else Q(0)
        assert value == expected
    assert sum(bool(value) for value in canonical_h6) == 32

    # Every partial-live survivor is a three-Pauli product; hence the exhaustive
    # search creates no additional irreducible six-port orbit.
    for assignment, solutions in safe.items():
        if all(line >= 0 for line in assignment):
            continue
        for scale in solutions:
            assert q_projective(reconstruct_tensor(assignment, scale)) in PRODUCT6

    assert core_count == 16
    assert product_count + core_count == sum(by_live_cards.values())
    print("  complete 49^3 normalized card space: PASS")
    print("  arbitrary complex scales via exact rational solution subspaces: PASS")
    print("  all-live survivors: 16 products + 16 H6 orbit tensors")
    print("  partial-live survivors: all products")
    print({
        "normalized_card_states": 49 ** 3,
        "safe_scale_lines_by_live_cards": dict(sorted(by_live_cards.items())),
        "product_scale_lines": product_count,
        "H6_scale_lines": core_count,
        "all_candidate_cards": dict(candidate_card_histogram),
    })
    print("V4 Q6 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
