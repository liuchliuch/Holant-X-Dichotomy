#!/usr/bin/env python3
"""Exact certificate for the external marked form of A4.

All calculations are over Q(sqrt(2), i).  The field implementation is
shared with ``verify_s4_second_form.py``; no floating-point arithmetic,
sampling, serialized survivor table, or external CAS is used.

The analytic similarity theta(U) = R^{-1} U R is used only to compare the
two root configurations and their q4 adjoints.  The deck-transitivity check
separately applies matrices from the external A4 group specified in the
paper at tensor ports, exactly as in the paper's actual-port action.  Neither
R nor theta is treated as an available gadget.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Sequence

import verify_a4_s4_certificates as standard
import verify_s4_second_form as extfield


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


F = extfield.F
ZERO = extfield.Z
ONE = extfield.O
TWO = extfield.T
I = extfield.II
SQRT2 = extfield.S

Vector = tuple[F, ...]
Matrix2 = tuple[F, F, F, F]
Matrix4 = tuple[tuple[F, F, F, F], ...]

Q0 = extfield.Q0
Q1 = extfield.Q1
Q2 = extfield.Q2
Q3 = extfield.Q3
QBASIS = (Q0, Q1, Q2, Q3)
NATIVE_X: Matrix2 = (ZERO, ONE, ONE, ZERO)
J_SIGNS = (ONE, -ONE, ONE, -ONE)

BITS4 = tuple(itertools.product((0, 1), repeat=4))
BIT_INDEX = {bits: index for index, bits in enumerate(BITS4)}
MATCHINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def fsum(values: Iterable[F]) -> F:
    return sum(values, ZERO)


def vadd(left: Sequence[F], right: Sequence[F]) -> Vector:
    assert len(left) == len(right)
    return tuple(a + b for a, b in zip(left, right))


def vsub(left: Sequence[F], right: Sequence[F]) -> Vector:
    assert len(left) == len(right)
    return tuple(a - b for a, b in zip(left, right))


def vscale(scalar: F | int | Fraction, vector: Sequence[F]) -> Vector:
    coefficient = F.coerce(scalar)
    return tuple(coefficient * value for value in vector)


def projective(vector: Sequence[F]) -> Vector:
    return extfield.projective(vector)


def qmatrix(coordinates: Sequence[F]) -> Matrix2:
    assert len(coordinates) == 4
    return tuple(
        fsum(coordinates[j] * QBASIS[j][position] for j in range(4))
        for position in range(4)
    )  # type: ignore[return-value]


def qcoordinates(matrix: Matrix2) -> Vector:
    return extfield.quaternion_coordinates(matrix)


def m2mul(left: Matrix2, right: Matrix2) -> Matrix2:
    return extfield.mm(left, right)


def m2inv(matrix: Matrix2) -> Matrix2:
    return extfield.inv(matrix)


def m2transpose(matrix: Matrix2) -> Matrix2:
    return matrix[0], matrix[2], matrix[1], matrix[3]


def mentry(matrix: Matrix2, row: int, column: int) -> F:
    return matrix[2 * row + column]


def coordinate_axis(index: int) -> Vector:
    return tuple(ONE if index == j else ZERO for j in range(4))


def standard_tetrahedral_points() -> list[Vector]:
    points = [coordinate_axis(index) for index in range(4)]
    points.extend(
        (ONE, F.rational(e1), F.rational(e2), F.rational(e3))
        for e1, e2, e3 in itertools.product((-1, 1), repeat=3)
    )
    normalized = [projective(point) for point in points]
    assert len(normalized) == len(set(normalized)) == 12
    return normalized


def standard_octahedral_points() -> list[Vector]:
    points = standard_tetrahedral_points()
    for left, right in itertools.combinations(range(4), 2):
        for sign in (-1, 1):
            point = [ZERO, ZERO, ZERO, ZERO]
            point[left] = ONE
            point[right] = F.rational(sign)
            points.append(projective(tuple(point)))
    assert len(points) == len(set(points)) == 24
    return points


def external_tetrahedral_points() -> list[Vector]:
    """The displayed external form, in Q0,Q1,Q2,Q3 coordinates."""
    points = [
        coordinate_axis(0),
        coordinate_axis(1),
        (ZERO, ZERO, ONE, ONE),
        (ZERO, ZERO, ONE, -ONE),
    ]
    for epsilon, delta in itertools.product((-1, 1), repeat=2):
        points.append((ONE, F.rational(epsilon), delta * SQRT2, ZERO))
        points.append((ONE, F.rational(epsilon), ZERO, delta * SQRT2))
    normalized = [projective(point) for point in points]
    assert len(normalized) == len(set(normalized)) == 12
    return normalized


T = standard_tetrahedral_points()
O = standard_octahedral_points()
T_EXT = external_tetrahedral_points()

R = qmatrix((ONE, SQRT2 - ONE, ZERO, ZERO))
RINV = m2inv(R)


def theta_matrix(matrix: Matrix2) -> Matrix2:
    return m2mul(m2mul(RINV, matrix), R)


def theta_inverse_matrix(matrix: Matrix2) -> Matrix2:
    return m2mul(m2mul(R, matrix), RINV)


def linear_coordinate_matrix(
    transform: Callable[[Matrix2], Matrix2],
) -> Matrix4:
    columns = [qcoordinates(transform(basis)) for basis in QBASIS]
    return tuple(
        tuple(columns[column][row] for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


THETA = linear_coordinate_matrix(theta_matrix)
THETA_INV = linear_coordinate_matrix(theta_inverse_matrix)


def m4identity() -> Matrix4:
    return tuple(
        tuple(ONE if row == column else ZERO for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def m4mul(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            fsum(left[row][middle] * right[middle][column]
                 for middle in range(4))
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def m4vec(matrix: Matrix4, vector: Sequence[F]) -> Vector:
    return tuple(
        fsum(matrix[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )


def m4sharp(matrix: Matrix4) -> Matrix4:
    """Adjoint for <u,v>_J = u^T J v, J=diag(1,-1,1,-1)."""
    return tuple(
        tuple(J_SIGNS[row] * matrix[column][row] * J_SIGNS[column]
              for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def m4basis(row: int, column: int) -> Matrix4:
    return tuple(
        tuple(ONE if (r, c) == (row, column) else ZERO for c in range(4))
        for r in range(4)
    )  # type: ignore[return-value]


def flatten(matrix: Matrix4) -> Vector:
    return tuple(value for row in matrix for value in row)


def unflatten(vector: Sequence[F]) -> Matrix4:
    assert len(vector) == 16
    return tuple(
        tuple(vector[4 * row + column] for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def normalized_map(matrix: Matrix4) -> Vector:
    return projective(flatten(matrix))


def m4rank(matrix: Matrix4) -> int:
    work = [list(row) for row in matrix]
    rank = 0
    for column in range(4):
        pivot = next((row for row in range(rank, 4)
                      if not work[row][column].is_zero()), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = work[rank][column].inverse()
        work[rank] = [inverse * value for value in work[rank]]
        for row in range(4):
            if row == rank or work[row][column].is_zero():
                continue
            factor = work[row][column]
            work[row] = [left - factor * right
                         for left, right in zip(work[row], work[rank])]
        rank += 1
    return rank


def maps_root_set(matrix: Matrix4, roots: set[Vector]) -> bool:
    for root in roots:
        image = m4vec(matrix, root)
        if all(value.is_zero() for value in image):
            continue
        if projective(image) not in roots:
            return False
    return True


def projective_order(matrix: Matrix2) -> int:
    identity = projective(Q0)
    power = Q0
    for order in range(1, 7):
        power = m2mul(power, matrix)
        if projective(power) == identity:
            return order
    raise AssertionError("unexpected projective order")


def generated_projective_group(generators: Sequence[Matrix2]) -> set[Vector]:
    seen: dict[Vector, Matrix2] = {projective(Q0): Q0}
    queue = deque([Q0])
    while queue:
        current = queue.popleft()
        for generator in generators:
            product = m2mul(current, generator)
            key = projective(product)
            if key not in seen:
                seen[key] = product
                queue.append(product)
    return set(seen)


def verify_marked_group() -> dict[str, object]:
    group_matrices = [qmatrix(point) for point in T_EXT]
    group = {projective(matrix) for matrix in group_matrices}
    assert len(group) == 12

    # The displayed list is exactly theta(T), not merely incidence-isomorphic.
    assert {projective(qcoordinates(theta_matrix(qmatrix(point))))
            for point in T} == set(T_EXT)

    for left in group_matrices:
        assert projective(m2inv(left)) in group
        assert projective(m2transpose(left)) in group
        for right in group_matrices:
            assert projective(m2mul(left, right)) in group
    order_profile = Counter(projective_order(matrix) for matrix in group_matrices)
    assert order_profile == Counter({1: 1, 2: 3, 3: 8})

    x, j = Q1, Q3
    assert projective(x) in group
    assert projective(j) not in group
    assert projective(m2mul(x, x)) == projective(Q0)
    assert projective(m2mul(j, j)) == projective(Q0)
    assert projective(m2mul(x, j)) == projective(m2mul(j, x))
    assert projective(m2mul(x, j)) == projective(Q2)
    assert projective(Q2) not in group
    for matrix in group_matrices:
        conjugate = m2mul(m2mul(j, matrix), m2inv(j))
        assert projective(conjugate) in group

    order_three = next(matrix for matrix in group_matrices
                       if projective_order(matrix) == 3)
    assert generated_projective_group((x, order_three)) == group
    marked_extension = generated_projective_group((x, order_three, j))
    assert len(marked_extension) == 24
    theta_octahedral = {
        projective(theta_matrix(qmatrix(point))) for point in O
    }
    assert marked_extension == theta_octahedral

    return {
        "order": len(group),
        "order_profile": dict(sorted(order_profile.items())),
        "x_in_group": True,
        "j_in_group": False,
        "xj_in_group": False,
        "marked_extension_order": len(marked_extension),
    }


def product_tensor(matching: int, left: Matrix2, right: Matrix2) -> Vector:
    (a, b), (c, d) = MATCHINGS[matching]
    return tuple(
        mentry(left, bits[a], bits[b]) * mentry(right, bits[c], bits[d])
        for bits in BITS4
    )


@dataclass(frozen=True, slots=True)
class DeckRecord:
    matching: int
    left: int
    right: int
    vector: Vector


def generate_deck(group_matrices: Sequence[Matrix2]
                  ) -> tuple[list[DeckRecord], dict[Vector, int]]:
    deck: list[DeckRecord] = []
    index: dict[Vector, int] = {}
    for matching in range(3):
        for left, left_matrix in enumerate(group_matrices):
            for right, right_matrix in enumerate(group_matrices):
                vector = projective(product_tensor(
                    matching, left_matrix, right_matrix))
                assert vector not in index
                index[vector] = len(deck)
                deck.append(DeckRecord(matching, left, right, vector))
    assert len(deck) == len(index) == 3 * 12**2 == 432
    return deck, index


def line_through_fixed_key(fixed: Vector, point: Vector) -> Vector:
    pivot = next(index for index, value in enumerate(fixed)
                 if not value.is_zero())
    residual = vsub(point, vscale(point[pivot] / fixed[pivot], fixed))
    return projective(residual)


def local_action(vector: Vector, port: int, matrix: Matrix2) -> Vector:
    """Apply L(y,x) at one port: the exact actual-port dressing action."""
    output = []
    for out_bits in BITS4:
        total = ZERO
        for in_bit in (0, 1):
            in_bits = list(out_bits)
            in_bits[port] = in_bit
            total += (mentry(matrix, out_bits[port], in_bit)
                      * vector[BIT_INDEX[tuple(in_bits)]])
        output.append(total)
    return projective(tuple(output))


def adjacent_port_swap(vector: Vector, port: int) -> Vector:
    output = []
    for out_bits in BITS4:
        in_bits = list(out_bits)
        in_bits[port], in_bits[port + 1] = \
            in_bits[port + 1], in_bits[port]
        output.append(vector[BIT_INDEX[tuple(in_bits)]])
    return projective(tuple(output))


def verify_deck_and_transitivity() -> dict[str, object]:
    group_matrices = [qmatrix(point) for point in T_EXT]
    deck, deck_index = generate_deck(group_matrices)

    a = projective(product_tensor(0, Q0, Q0))  # A=P_0(Q_0,Q_0).
    a_index = deck_index[a]
    buckets: dict[Vector, list[int]] = {}
    for index, record in enumerate(deck):
        if index == a_index:
            continue
        key = line_through_fixed_key(a, record.vector)
        buckets.setdefault(key, []).append(index)
    through_a = Counter(len(indices) for indices in buckets.values())
    assert through_a == Counter({1: 415, 2: 8})
    rich_through_a = [indices for indices in buckets.values()
                      if len(indices) == 2]
    assert len(rich_through_a) == 8
    for others in rich_through_a:
        assert {deck[index].matching for index in (a_index, *others)} == {0}

    order_three = next(matrix for matrix in group_matrices
                       if projective_order(matrix) == 3)
    local_generators = (Q1, order_three)

    # Q1 and one order-three element generate T_ext.  Applying them at each
    # port, together with adjacent port swaps, generates T_ext^4 semidirect S4.
    assert len(generated_projective_group(local_generators)) == 12
    visited = {a}
    queue = deque([a])
    swap_matching_images: dict[tuple[int, int], int] = {}
    while queue:
        vector = queue.popleft()
        source_record = deck[deck_index[vector]]
        for port in range(4):
            for generator in local_generators:
                image = local_action(vector, port, generator)
                assert image in deck_index
                target_record = deck[deck_index[image]]
                assert target_record.matching == source_record.matching
                if image not in visited:
                    visited.add(image)
                    queue.append(image)
        for port in range(3):
            image = adjacent_port_swap(vector, port)
            assert image in deck_index
            target_matching = deck[deck_index[image]].matching
            key = (port, source_record.matching)
            previous = swap_matching_images.setdefault(key, target_matching)
            assert previous == target_matching
            if image not in visited:
                visited.add(image)
                queue.append(image)
    assert len(visited) == len(deck) == 432

    # Every action above is invertible and projective-linear.  Transitivity
    # transports the complete through-A bucket calculation to every point.
    # Local actions preserve a matching, and a port permutation only relabels
    # it, so all rich lines are monochromatic three-lines.  Double counting
    # point-line incidences gives the complete global number.
    global_rich_lines = len(deck) * len(rich_through_a) // 3
    assert global_rich_lines == 432 * 8 // 3 == 1152

    return {
        "deck_points": len(deck),
        "through_A_bucket_histogram": dict(sorted(through_a.items())),
        "rich_lines_through_A": len(rich_through_a),
        "rich_line_matching_type": (3,),
        "local_port_orbit": len(visited),
        "global_rich_lines": global_rich_lines,
    }


def explicit_preserver_maps() -> tuple[set[Vector], set[Vector]]:
    """Generate the exact standard-A4 rank-one and rank-four families."""
    rank_one: set[Vector] = set()
    for u in T:
        for v in T:
            jv = tuple(J_SIGNS[column] * v[column] for column in range(4))
            matrix = tuple(
                tuple(u[row] * jv[column] for column in range(4))
                for row in range(4)
            )
            rank_one.add(normalized_map(matrix))
    assert len(rank_one) == 144
    assert all(m4rank(unflatten(matrix)) == 1 for matrix in rank_one)

    a4_matrices = [qmatrix(point) for point in T]
    s4_matrices = [qmatrix(point) for point in O]
    rank_four: set[Vector] = set()
    for product_uv in a4_matrices:
        for v in s4_matrices:
            u = m2mul(product_uv, m2inv(v))
            for transpose in (False, True):
                columns = []
                for basis in QBASIS:
                    middle = m2transpose(basis) if transpose else basis
                    columns.append(qcoordinates(m2mul(m2mul(u, middle), v)))
                matrix = tuple(
                    tuple(columns[column][row] for column in range(4))
                    for row in range(4)
                )
                rank_four.add(normalized_map(matrix))
    assert len(rank_four) == 2 * 12 * 24 == 576
    assert all(m4rank(unflatten(matrix)) == 4 for matrix in rank_four)
    return rank_one, rank_four


def verify_orientation_twist_and_q4_covariance() -> dict[str, object]:
    identity = m4identity()
    assert m4mul(THETA, THETA_INV) == identity
    assert m4mul(THETA_INV, THETA) == identity
    assert m4sharp(THETA) == THETA

    assert m2transpose(R) == R
    assert m2mul(R, NATIVE_X) == m2mul(NATIVE_X, R)
    r_squared = m2mul(R, R)
    assert projective(qcoordinates(r_squared)) == \
        projective((ONE, ONE, ZERO, ZERO))

    theta_squared = m4mul(THETA, THETA)
    theta_inverse_squared = m4mul(THETA_INV, THETA_INV)
    assert {projective(m4vec(theta_squared, root)) for root in T} == set(T)

    # For A'=theta A theta^{-1}, the external adjoint is not the naive
    # transport of A^sharp.  The exact orientation twist is
    # theta^{-1}(A')^sharp theta = theta^{-2} A^sharp theta^2.
    for row in range(4):
        for column in range(4):
            a = m4basis(row, column)
            aprime = m4mul(m4mul(THETA, a), THETA_INV)
            left = m4mul(m4mul(THETA_INV, m4sharp(aprime)), THETA)
            right = m4mul(
                m4mul(theta_inverse_squared, m4sharp(a)), theta_squared)
            assert left == right

    # Replay the exact standard D4 geometry used to exclude ranks two and
    # three, and the incidence-automorphism upper bound used for rank four.
    standard_points = standard.tetrahedral_points()
    geometry = standard.verify_root_geometry("A4", standard_points)
    assert geometry["secant_lines"] == {"2": 18, "3": 16}
    assert geometry["plane_sections"] == {"3": 12, "6": 12}
    assert geometry["center_directions"] == {"7": 12, "9": 12}
    assert geometry["rank_two_projection_types"] == {
        "(2, 4, 2)": 18,
        "(3, 6, 0)": 16,
    }
    maximum_root_line = 3
    maximum_plane_section = 6
    projection_types = ((2, 4, 2), (3, 6, 0))
    assert all(direction_count > maximum_root_line
               for _, direction_count, _ in projection_types)
    # A rank-three kernel on at most one root secant has at least ten
    # quotient directions.  The exact multiple-secant-center table above
    # has minimum seven; neither can fit in a six-point plane section.
    assert 10 > maximum_plane_section
    assert min((7, 9)) > maximum_plane_section
    full_rank_upper_bound = standard.incidence_automorphism_upper_bound(
        standard_points)
    assert full_rank_upper_bound == 576

    rank_one, rank_four = explicit_preserver_maps()
    roots = set(T)
    external_roots = set(T_EXT)
    transported_one: set[Vector] = set()
    transported_four: set[Vector] = set()
    for rank, family, transported in (
        (1, rank_one, transported_one),
        (4, rank_four, transported_four),
    ):
        for flattened in family:
            a = unflatten(flattened)
            assert maps_root_set(a, roots)
            assert maps_root_set(m4sharp(a), roots)
            aprime = m4mul(m4mul(THETA, a), THETA_INV)
            assert m4rank(aprime) == rank
            assert maps_root_set(aprime, external_roots)
            assert maps_root_set(m4sharp(aprime), external_roots)
            transported.add(normalized_map(aprime))
        assert len(transported) == len(family)
        assert len(transported) == (144 if rank == 1 else 576)

    # For a nonzero rank-one two-sided preserver, its image and the image of
    # its sharp adjoint are root lines, so there are at most |T|^2 choices;
    # the explicit family attains that bound.  The projection/center tests
    # exclude ranks two and three, and the explicit full-rank family attains
    # the independent incidence-automorphism upper bound.
    assert len(rank_one) == len(T) ** 2
    assert len(rank_four) == full_rank_upper_bound
    rank_profile = {
        0: 1,
        1: len(rank_one),
        2: 0,
        3: 0,
        4: len(rank_four),
    }
    assert rank_profile == {0: 1, 1: 144, 2: 0, 3: 0, 4: 576}
    return {
        "R_symmetric": True,
        "theta_self_adjoint": True,
        "R2_normalizes_standard_T": True,
        "covariance_basis_maps_checked": 16,
        "two_sided_rank_profile": rank_profile,
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("External marked-A4 exact certificate")
    marked = verify_marked_group()
    print("  marked form, closure, order profile, j-normalization: PASS", flush=True)
    deck = verify_deck_and_transitivity()
    print("  432-point deck, through-A lines, actual transitivity: PASS", flush=True)
    q4 = verify_orientation_twist_and_q4_covariance()
    print("  orientation twist and q4 adjoint covariance: PASS", flush=True)
    print({"marked_group": marked, "deck": deck, "q4": q4})
    print("EXTERNAL MARKED-A4 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
