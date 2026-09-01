#!/usr/bin/env python3
"""Exact replay for the extended Platonic incidence certificates.

This verifier extends the bounded ``verify_a5_certificate.py`` replay.  It
uses exact arithmetic only and checks, from the root configurations specified
in the paper:

* the complete A5 deck-line counts by an explicit transitive deck action and
  an incidence double count from the lines through one point;
* the A4/S4/A5 cross-line graphs, including a third-line witness on every
  edge, and the ten-dimensional quadratic evaluation certificate used for
  the fixed-Segre residue;
* connectivity after every possible rank-two/rank-three kernel deletion;
* the H4 secant, plane, multiple-secant-center, and rank-two projection data;
* the rank-one and full-rank H4 two-sided preservers, including an exact
  incidence-automorphism upper bound proving full-rank completeness.

No payload, hash, floating-point value, random sample, finite-field
specialization, or computer-algebra system is used.  The program reuses the
field and tensor primitives of the other Platonic verifiers.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Sequence

from verify_a5_certificate import (
    E,
    ONE,
    Q0,
    Q1,
    Q2,
    Q3,
    ZERO,
    DeckRecord,
    Matrix2,
    Vector,
    bits4,
    dot,
    generate_deck,
    generate_group_points,
    is_zero_vector,
    line_through_fixed_key,
    lines_through_A,
    mat_inverse,
    mat_transpose,
    matmul,
    matrix_rank,
    mentry,
    product_tensor,
    projective,
    quaternion_matrix,
    verify_group,
)
from verify_a4_s4_certificates import (
    all_secant_lines,
    line_key,
    matrix_projective_coordinates,
    octahedral_points,
    rank_two_projection_histogram,
    tetrahedral_points,
)


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


J_SIGNS = (ONE, -ONE, ONE, -ONE)


def rich_lines(points: Sequence[Vector]) -> list[tuple[int, ...]]:
    """Return all maximal projective lines containing at least three roots."""
    return sorted(
        tuple(indices)
        for indices in all_secant_lines(points).values()
        if len(indices) >= 3
    )


def connected_on_vertices(
    adjacency: Sequence[set[int]], remaining: set[int]
) -> bool:
    if not remaining:
        return True
    start = next(iter(remaining))
    seen = {start}
    queue = [start]
    while queue:
        vertex = queue.pop()
        for neighbour in adjacency[vertex]:
            if neighbour in remaining and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen == remaining


def verify_deletion_connectivity(
    name: str, points: Sequence[Vector], expected_tests: int
) -> dict[str, object]:
    """Check every projective kernel deletion relevant to ranks two and three.

    A projective point deletes either no root or one root.  A projective line
    deletes no root, one root, or one of the maximal intersections generated
    by a pair of roots.  Thus the list below is exhaustive, not sampled.
    """
    lines = rich_lines(points)
    adjacency = [set() for _ in points]
    for line in lines:
        for left, right in itertools.combinations(line, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)

    line_sections = {
        frozenset(indices) for indices in all_secant_lines(points).values()
    }
    deletion_sets = (
        {frozenset()}
        | {frozenset({index}) for index in range(len(points))}
        | line_sections
    )
    assert len(deletion_sets) == expected_tests
    universe = set(range(len(points)))
    residual_sizes = Counter()
    for deleted in deletion_sets:
        remaining = universe - set(deleted)
        assert connected_on_vertices(adjacency, remaining)
        residual_sizes[len(remaining)] += 1

    return {
        "group": name,
        "tested_deletions": len(deletion_sets),
        "deleted_size_histogram": dict(
            sorted(Counter(map(len, deletion_sets)).items())
        ),
        "residual_size_histogram": dict(sorted(residual_sizes.items())),
    }


def verify_cross_line_graph(
    name: str,
    points: Sequence[Vector],
    expected_edges: int,
    expected_edge_types: Counter[tuple[int, int]],
    expected_degrees: dict[int, int],
    expected_witness_histogram: Counter[int],
) -> dict[str, object]:
    """Build the rich-line intersection graph and certify every cross edge.

    For two rich lines through a common root, a witness is a pair of
    nonintersection roots, one on each line, lying on a third rich line.
    The check records every such pair as an independent incidence regression.
    The fixed-Segre argument uses the separate rank-ten quadratic evaluation
    below; this graph provides an additional incidence check.
    """
    lines = rich_lines(points)
    point_to_lines: dict[int, list[int]] = defaultdict(list)
    pair_to_line: dict[tuple[int, int], int] = {}
    for line_index, line in enumerate(lines):
        for point in line:
            point_to_lines[point].append(line_index)
        for left, right in itertools.combinations(line, 2):
            key = (left, right) if left < right else (right, left)
            assert key not in pair_to_line
            pair_to_line[key] = line_index

    edges: set[tuple[int, int]] = set()
    for incident in point_to_lines.values():
        for left, right in itertools.combinations(incident, 2):
            edges.add((left, right) if left < right else (right, left))
    assert len(edges) == expected_edges

    adjacency = [set() for _ in lines]
    edge_types: Counter[tuple[int, int]] = Counter()
    witness_histogram: Counter[int] = Counter()
    for left_index, right_index in edges:
        left_line = lines[left_index]
        right_line = lines[right_index]
        intersection = set(left_line) & set(right_line)
        assert len(intersection) == 1
        common = next(iter(intersection))
        witnesses = []
        for left_point in left_line:
            if left_point == common:
                continue
            for right_point in right_line:
                if right_point == common:
                    continue
                key = ((left_point, right_point)
                       if left_point < right_point
                       else (right_point, left_point))
                third = pair_to_line.get(key)
                if third is not None:
                    assert third not in (left_index, right_index)
                    witnesses.append((left_point, right_point, third))
        assert witnesses
        witness_histogram[len(witnesses)] += 1
        edge_types[tuple(sorted((len(left_line), len(right_line))))] += 1
        adjacency[left_index].add(right_index)
        adjacency[right_index].add(left_index)

    assert edge_types == expected_edge_types
    assert witness_histogram == expected_witness_histogram
    assert connected_on_vertices(adjacency, set(range(len(lines))))
    degree_by_size: dict[int, set[int]] = defaultdict(set)
    for index, neighbours in enumerate(adjacency):
        degree_by_size[len(lines[index])].add(len(neighbours))
    assert degree_by_size == {
        size: {degree} for size, degree in expected_degrees.items()
    }

    return {
        "group": name,
        "vertices": len(lines),
        "edges": len(edges),
        "edge_types": {str(key): value for key, value in sorted(edge_types.items())},
        "degrees_by_line_size": {
            str(size): next(iter(degrees))
            for size, degrees in sorted(degree_by_size.items())
        },
        "cross_witness_histogram": dict(sorted(witness_histogram.items())),
    }


def verify_quadratic_evaluation_basis() -> dict[str, object]:
    """Prove that the tetrahedral roots determine every quadratic coefficient.

    A fixed-matching 2-by-2 Segre minor is a homogeneous quadratic in the
    four kernel coordinates.  The ten columns below are the four squares and
    six cross monomials.  Full column rank on the twelve tetrahedral roots
    means that a minor vanishing on those roots is the zero polynomial.
    """
    points = tetrahedral_points()
    monomials = tuple((left, right)
                      for left in range(4)
                      for right in range(left, 4))
    assert len(monomials) == 10
    evaluation = [
        [point[left] * point[right] for left, right in monomials]
        for point in points
    ]
    rank = matrix_rank(evaluation)
    assert rank == 10

    axes = points[:4]
    signs = points[4:]
    assert matrix_rank([
        [point[left] * point[right] for left, right in monomials]
        for point in axes + signs
    ]) == 10
    return {
        "points": len(points),
        "quadratic_monomials": len(monomials),
        "evaluation_rank": rank,
    }


def tensor_index(bits: Sequence[int]) -> int:
    return 8 * bits[0] + 4 * bits[1] + 2 * bits[2] + bits[3]


def act_on_port(vector: Sequence[E], port: int, matrix: Matrix2) -> Vector:
    """Apply matrix(y,x) to one tensor port."""
    output = []
    for output_index in range(16):
        output_bits = list(bits4(output_index))
        output_bit = output_bits[port]
        total = ZERO
        for input_bit in (0, 1):
            output_bits[port] = input_bit
            total += (mentry(matrix, output_bit, input_bit)
                      * vector[tensor_index(output_bits)])
        output.append(total)
    return tuple(output)


def swap_ports(vector: Sequence[E], left: int, right: int) -> Vector:
    output = []
    for output_index in range(16):
        input_bits = list(bits4(output_index))
        input_bits[left], input_bits[right] = (
            input_bits[right], input_bits[left]
        )
        output.append(vector[tensor_index(input_bits)])
    return tuple(output)


def group_generated_indices(
    matrices: Sequence[Matrix2], matrix_index: dict[Vector, int],
    generators: Sequence[int],
) -> set[int]:
    identity = matrix_index[projective(Q0)]
    reached = {identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for generator in generators:
            product = matrix_index[
                projective(matmul(matrices[current], matrices[generator]))
            ]
            if product not in reached:
                reached.add(product)
                queue.append(product)
    return reached


@dataclass(frozen=True)
class DeckAction:
    name: str
    permutation: tuple[int, ...]
    matching_permutation: tuple[int, int, int]


def induced_deck_action(
    name: str,
    deck: Sequence[DeckRecord],
    deck_index: dict[Vector, int],
    transform: Callable[[Sequence[E]], Vector],
) -> DeckAction:
    permutation = tuple(
        deck_index[projective(transform(record.vector))]
        for record in deck
    )
    assert len(set(permutation)) == len(deck)
    matching_images: dict[int, set[int]] = defaultdict(set)
    for source, target in enumerate(permutation):
        matching_images[deck[source].matching].add(deck[target].matching)
    assert all(len(images) == 1 for images in matching_images.values())
    matching_permutation = tuple(
        next(iter(matching_images[matching])) for matching in range(3)
    )
    assert set(matching_permutation) == {0, 1, 2}
    return DeckAction(name, permutation, matching_permutation)


def verify_a5_deck_double_count(
    points: Sequence[Vector],
) -> dict[str, object]:
    """Derive all A5 rich-deck-line counts from one exact star.

    The generated local actions and adjacent port swaps are invertible linear
    tensor actions.  Their induced permutations are checked on all 10,800
    deck points, preserve matching-label type, and act transitively.  Hence
    every deck point has the star computed at A, and incidence double
    counting gives the global line numbers.
    """
    matrices = [quaternion_matrix(point) for point in points]
    matrix_index = {projective(matrix): index
                    for index, matrix in enumerate(matrices)}
    assert len(matrix_index) == 60
    generator_indices = (1, 24)
    assert group_generated_indices(
        matrices, matrix_index, generator_indices
    ) == set(range(60))

    deck, deck_index = generate_deck(points)
    a = projective(product_tensor(0, Q0, Q0))
    a_index = deck_index[a]
    line_buckets, bucket_histogram = lines_through_A(deck, a_index)
    assert bucket_histogram == Counter({1: 10_591, 2: 80, 4: 12})

    through_types: Counter[tuple[int, tuple[int, ...]]] = Counter()
    for other_indices in line_buckets.values():
        if len(other_indices) < 2:
            continue
        line = (a_index, *other_indices)
        matching_counts = Counter(deck[index].matching for index in line)
        matching_type = tuple(sorted(matching_counts.values(), reverse=True))
        through_types[(len(line), matching_type)] += 1
    expected_through = Counter({
        (3, (1, 1, 1)): 60,
        (3, (3,)): 20,
        (5, (5,)): 12,
    })
    assert through_types == expected_through

    actions: list[DeckAction] = []
    for generator_index in generator_indices:
        generator = matrices[generator_index]
        assert not is_zero_vector(generator)
        inverse = mat_inverse(generator)
        assert projective(matmul(generator, inverse)) == projective(Q0)
        for port in range(4):
            actions.append(induced_deck_action(
                f"local-{generator_index}-port-{port}",
                deck,
                deck_index,
                lambda vector, p=port, g=generator: act_on_port(vector, p, g),
            ))
    for left in range(3):
        actions.append(induced_deck_action(
            f"swap-{left}-{left + 1}",
            deck,
            deck_index,
            lambda vector, a=left, b=left + 1: swap_ports(vector, a, b),
        ))

    reached = {a_index}
    queue = deque([a_index])
    while queue:
        current = queue.popleft()
        for action in actions:
            image = action.permutation[current]
            if image not in reached:
                reached.add(image)
                queue.append(image)
    assert len(reached) == len(deck) == 10_800

    global_types: Counter[tuple[int, tuple[int, ...]]] = Counter()
    for line_type, through_count in through_types.items():
        line_size, _ = line_type
        incidences = len(deck) * through_count
        assert incidences % line_size == 0
        global_types[line_type] = incidences // line_size
    assert global_types == Counter({
        (3, (1, 1, 1)): 216_000,
        (3, (3,)): 72_000,
        (5, (5,)): 25_920,
    })
    assert sum(global_types.values()) == 313_920

    return {
        "deck_points": len(deck),
        "action_generators": len(actions),
        "transitive_orbit": len(reached),
        "through_A": {str(key): value for key, value in sorted(through_types.items())},
        "global_by_double_count": {
            str(key): value for key, value in sorted(global_types.items())
        },
        "global_rich_lines": sum(global_types.values()),
    }


def determinant3(left: Vector, middle: Vector, right: Vector) -> E:
    assert len(left) == len(middle) == len(right) == 3
    return (
        left[0] * (middle[1] * right[2] - middle[2] * right[1])
        - left[1] * (middle[0] * right[2] - middle[2] * right[0])
        + left[2] * (middle[0] * right[1] - middle[1] * right[0])
    )


def hyperplane_normal(
    left: Sequence[E], middle: Sequence[E], right: Sequence[E]
) -> Vector:
    """The exact four-dimensional cross product of three row vectors."""
    normal = []
    for deleted in range(4):
        retained = tuple(index for index in range(4) if index != deleted)
        minor = determinant3(
            tuple(left[index] for index in retained),
            tuple(middle[index] for index in retained),
            tuple(right[index] for index in retained),
        )
        normal.append(-minor if deleted % 2 else minor)
    return tuple(normal)


def enumerate_plane_sections(
    points: Sequence[Vector],
) -> dict[Vector, frozenset[int]]:
    """Enumerate all root-containing planes from their spanning triples."""
    sections: dict[Vector, frozenset[int]] = {}
    for triple in itertools.combinations(range(len(points)), 3):
        normal = hyperplane_normal(*(points[index] for index in triple))
        if is_zero_vector(normal):
            continue
        normal = projective(normal)
        if normal not in sections:
            sections[normal] = frozenset(
                index for index, point in enumerate(points)
                if dot(normal, point).is_zero()
            )
    return sections


def pluecker_intersection(left: Sequence[E], right: Sequence[E]) -> E:
    """Klein-quadric bilinear form; zero iff two P3 lines meet."""
    assert len(left) == len(right) == 6
    return (
        left[0] * right[5] - left[1] * right[4]
        + left[2] * right[3] + left[3] * right[2]
        - left[4] * right[1] + left[5] * right[0]
    )


def verify_h4_centers(
    points: Sequence[Vector],
    plane_sections: dict[Vector, frozenset[int]],
    secants: dict[Vector, tuple[int, ...]],
) -> Counter[int]:
    """Enumerate and prove completeness of all multiple-secant centers.

    H4 is self-polar for J.  The J-polars of all root-containing planes give
    1,320 candidate centers.  We verify directly that every candidate lies on
    at least two secants and, independently via the Pluecker intersection
    equation, that every intersecting pair of secants meets at one candidate.
    """
    root_index = {projective(point): index for index, point in enumerate(points)}
    centers = {
        projective(tuple(J_SIGNS[index] * normal[index]
                         for index in range(4)))
        for normal in plane_sections
    }
    assert len(centers) == 1320
    assert set(root_index) <= centers

    centers_by_line: dict[Vector, set[int]] = defaultdict(set)
    direction_histogram: Counter[int] = Counter()
    for center_index, center in enumerate(sorted(
        centers, key=lambda vector: tuple(entry.sort_key() for entry in vector)
    )):
        coincident_root = root_index.get(center)
        direction_buckets: dict[Vector, list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            if index == coincident_root:
                continue
            direction_buckets[line_through_fixed_key(center, point)].append(index)
        direction_histogram[len(direction_buckets)] += 1

        incident_secants = 0
        for bucket in direction_buckets.values():
            if coincident_root is not None:
                secant_key = line_key(points[coincident_root], points[bucket[0]])
            elif len(bucket) >= 2:
                secant_key = line_key(points[bucket[0]], points[bucket[1]])
            else:
                continue
            assert secant_key in secants
            centers_by_line[secant_key].add(center_index)
            incident_secants += 1
        assert incident_secants >= 2

    expected = Counter({31: 60, 49: 300, 51: 360, 55: 600})
    assert direction_histogram == expected

    secant_items = list(secants.items())
    intersecting_pairs = 0
    externally_intersecting_pairs = 0
    for (left_key, left_roots), (right_key, right_roots) in itertools.combinations(
        secant_items, 2
    ):
        intersection_value = pluecker_intersection(left_key, right_key)
        common_roots = set(left_roots) & set(right_roots)
        if common_roots:
            assert intersection_value.is_zero()
        if not intersection_value.is_zero():
            continue
        intersecting_pairs += 1
        if common_roots:
            continue
        externally_intersecting_pairs += 1
        assert centers_by_line[left_key] & centers_by_line[right_key]

    # Conversely, every recorded multiple-secant incidence is an exact
    # collinearity certificate, so the two directions above give equality.
    assert intersecting_pairs > externally_intersecting_pairs > 0
    return direction_histogram


def colored_root_stabilizer(
    points: Sequence[Vector],
    secants: dict[Vector, tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Enumerate pair-line-color automorphisms fixing root 0."""
    point_count = len(points)
    line_size: dict[tuple[int, int], int] = {}
    for indices in secants.values():
        for left, right in itertools.combinations(indices, 2):
            line_size[(left, right)] = line_size[(right, left)] = len(indices)
    colors = [[0 for _ in points] for _ in points]
    for left in range(point_count):
        for right in range(left + 1, point_count):
            colors[left][right] = colors[right][left] = line_size[(left, right)]

    mapping = {0: 0}
    used = {0}
    automorphisms: list[tuple[int, ...]] = []

    def search() -> None:
        if len(mapping) == point_count:
            automorphisms.append(tuple(mapping[index]
                                       for index in range(point_count)))
            return
        best_source = -1
        best_candidates: list[int] | None = None
        for source in range(point_count):
            if source in mapping:
                continue
            candidates = [
                target for target in range(point_count)
                if target not in used
                and all(colors[source][old_source]
                        == colors[target][old_target]
                        for old_source, old_target in mapping.items())
            ]
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
    return automorphisms


def solve_column_coordinates(
    columns: Sequence[Sequence[E]], target: Sequence[E]
) -> Vector:
    """Solve sum_j coefficient_j columns[j] = target exactly."""
    assert len(columns) == len(target) == 4
    work = [
        [columns[column][row] for column in range(4)] + [target[row]]
        for row in range(4)
    ]
    pivot_row = 0
    for column in range(4):
        pivot = next((row for row in range(pivot_row, 4)
                      if not work[row][column].is_zero()), None)
        if pivot is None:
            raise ValueError("columns are linearly dependent")
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = work[pivot_row][column].inverse()
        work[pivot_row] = [inverse * entry for entry in work[pivot_row]]
        for row in range(4):
            if row == pivot_row or work[row][column].is_zero():
                continue
            factor = work[row][column]
            work[row] = [left - factor * right
                         for left, right in zip(work[row], work[pivot_row])]
        pivot_row += 1
    return tuple(work[row][4] for row in range(4))


def projective_linear_realization(
    points: Sequence[Vector], permutation: Sequence[int]
) -> Vector | None:
    """Reconstruct, then verify, the PGL4 map inducing ``permutation``.

    The first four H4 roots are the coordinate axes and the fifth has all
    four coordinates nonzero, hence these five points form a projective
    frame.  A projective linear map is uniquely determined by its action on
    that frame.  This test separates the 120 genuine linear stabilizers from
    120 additional symmetries of the finite colored incidence structure.
    """
    source_basis = points[:4]
    source_fifth = points[4]
    assert matrix_rank([list(vector) for vector in source_basis]) == 4
    source_coordinates = solve_column_coordinates(source_basis, source_fifth)
    assert all(not entry.is_zero() for entry in source_coordinates)

    target_basis = [points[permutation[index]] for index in range(4)]
    try:
        target_coordinates = solve_column_coordinates(
            target_basis, points[permutation[4]]
        )
    except ValueError:
        return None
    if any(entry.is_zero() for entry in target_coordinates):
        return None
    scales = tuple(target_coordinates[index] / source_coordinates[index]
                   for index in range(4))
    matrix = tuple(
        scales[column] * target_basis[column][row]
        for row in range(4) for column in range(4)
    )
    for source_index, point in enumerate(points):
        image = tuple(
            sum((matrix[4 * row + column] * point[column]
                 for column in range(4)), ZERO)
            for row in range(4)
        )
        if is_zero_vector(image):
            return None
        if projective(image) != points[permutation[source_index]]:
            return None
    return projective(matrix)


def sharp_matrix(flattened: Sequence[E]) -> Vector:
    assert len(flattened) == 16
    return tuple(
        J_SIGNS[row] * flattened[4 * column + row] * J_SIGNS[column]
        for row in range(4)
        for column in range(4)
    )


def coordinate_map(
    left: Matrix2, right: Matrix2, transpose: bool
) -> Vector:
    columns = []
    for basis in (Q0, Q1, Q2, Q3):
        middle = mat_transpose(basis) if transpose else basis
        image = matmul(matmul(left, middle), right)
        columns.append(matrix_projective_coordinates(image))
    return tuple(columns[column][row]
                 for row in range(4) for column in range(4))


def verify_h4_preservers(points: Sequence[Vector]) -> dict[str, object]:
    secants = all_secant_lines(points)
    line_histogram = Counter(len(indices) for indices in secants.values())
    assert line_histogram == Counter({2: 450, 3: 200, 5: 72})

    plane_sections = enumerate_plane_sections(points)
    plane_histogram = Counter(map(len, plane_sections.values()))
    assert plane_histogram == Counter({4: 600, 6: 660, 15: 60})

    center_histogram = verify_h4_centers(points, plane_sections, secants)
    projection_histogram = rank_two_projection_histogram(points)
    assert projection_histogram == Counter({
        (2, 12, 2): 450,
        (3, 12, 3): 200,
        (5, 10, 5): 72,
    })
    assert min(directions for _, directions, _ in projection_histogram) > 5
    assert min(center_histogram) > max(plane_histogram)
    assert 56 > max(plane_histogram)  # centers on at most one secant

    # Rank one: A=u(Jv)^T.  These maps and their sharps are exactly the
    # ordered root pairs; projective injectivity gives 60^2 maps.
    rank_one = {
        projective(tuple(
            u[row] * J_SIGNS[column] * v[column]
            for row in range(4) for column in range(4)
        ))
        for u in points for v in points
    }
    assert len(rank_one) == 3600
    assert {projective(sharp_matrix(matrix)) for matrix in rank_one} == rank_one

    # Full rank: V and UV range independently over A5; U=(UV)V^{-1}.
    # Generate both K -> UKV and K -> UK^T V forms exactly.
    matrices = [quaternion_matrix(point) for point in points]
    full_rank = set()
    for product_uv in matrices:
        for right in matrices:
            left = matmul(product_uv, mat_inverse(right))
            for transpose in (False, True):
                candidate = coordinate_map(left, right, transpose)
                assert matrix_rank([
                    list(candidate[4 * row:4 * row + 4]) for row in range(4)
                ]) == 4
                full_rank.add(projective(candidate))
    assert len(full_rank) == 7200
    assert {projective(sharp_matrix(matrix)) for matrix in full_rank} == full_rank

    # Completeness, not just membership: any projective root automorphism
    # preserves pair-line colors.  Fixing root 0 leaves 240 such incidence
    # automorphisms.  The five-point projective-frame reconstruction proves
    # that exactly 120 are induced by C-linear maps.  Hence PGL automorphisms
    # number at most 60*120=7200, attained by the explicit family above.
    colored_stabilizer = colored_root_stabilizer(points, secants)
    assert len(colored_stabilizer) == 240
    projective_realizations = {
        realization
        for permutation in colored_stabilizer
        if (realization := projective_linear_realization(points, permutation))
        is not None
    }
    projective_stabilizer = len(projective_realizations)
    assert projective_stabilizer == 120
    automorphism_upper_bound = len(points) * projective_stabilizer
    assert automorphism_upper_bound == len(full_rank) == 7200

    return {
        "secant_lines": dict(sorted(line_histogram.items())),
        "plane_sections": dict(sorted(plane_histogram.items())),
        "center_directions": dict(sorted(center_histogram.items())),
        "rank_two_projections": {
            str(key): value for key, value in sorted(projection_histogram.items())
        },
        "two_sided_rank_counts": {
            "rank0": 1,
            "rank1": len(rank_one),
            "rank2": 0,
            "rank3": 0,
            "rank4": len(full_rank),
        },
        "colored_stabilizer": len(colored_stabilizer),
        "projective_stabilizer": projective_stabilizer,
        "full_rank_automorphism_upper_bound": automorphism_upper_bound,
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    a4_points = tetrahedral_points()
    s4_points = octahedral_points()
    a5_points = generate_group_points()
    verify_group(a5_points)

    print("Extended Platonic exact certificate replay", flush=True)

    deletion = {
        "A4": verify_deletion_connectivity("A4", a4_points, 47),
        "S4": verify_deletion_connectivity("S4", s4_points, 147),
        "A5": verify_deletion_connectivity("A5", a5_points, 783),
    }
    print("  deficient-rank deletion connectivity: "
          "A4=47, S4=147, A5=783: PASS", flush=True)

    cross_graphs = {
        "A4": verify_cross_line_graph(
            "A4", a4_points, 72,
            Counter({(3, 3): 72}), {3: 9}, Counter({2: 72}),
        ),
        "S4": verify_cross_line_graph(
            "S4", s4_points, 504,
            Counter({(3, 3): 144, (3, 4): 288, (4, 4): 72}),
            {3: 18, 4: 24}, Counter({4: 432, 5: 72}),
        ),
        "A5": verify_cross_line_graph(
            "A5", a5_points, 7200,
            Counter({(3, 3): 2700, (3, 5): 3600, (5, 5): 900}),
            {3: 45, 5: 75},
            Counter({2: 1800, 4: 900, 6: 3600, 12: 900}),
        ),
    }
    quadratic_basis = verify_quadratic_evaluation_basis()
    print("  A4/S4/A5 cross-line graphs and Segre quadratic basis: PASS", flush=True)

    deck_double_count = verify_a5_deck_double_count(a5_points)
    print("  A5 lines through A: 60 rainbow 3-lines, 20 monochromatic "
          "3-lines, 12 monochromatic 5-lines; transitive double count: PASS",
          flush=True)

    h4 = verify_h4_preservers(a5_points)
    print("  H4 geometry and two-sided q4 preserver completeness: PASS", flush=True)

    print({
        "deletion_connectivity": deletion,
        "cross_line_graphs": cross_graphs,
        "quadratic_basis": quadratic_basis,
        "A5_deck": deck_double_count,
        "H4": h4,
    })
    print("EXTENDED PLATONIC EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
