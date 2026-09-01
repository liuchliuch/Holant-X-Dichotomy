#!/usr/bin/env python3
"""Exact replay of the affine endpoint-nondegenerate quaternary certificate.

The program constructs every projective affine signature on four Boolean
variables, applies every one of the eighteen edge-compatible basis cores,
and checks the endpoint-nondegenerate and fourth-root residues used in the
paper.
It uses only exact arithmetic in Z[i,sqrt(2)] and Q(i,sqrt(2)).  There are
no stored survivor lists, hashes, random choices, floating-point operations,
or finite-field specializations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, product
from math import comb

if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


# Elements of Z[sqrt(2),i] are tuples for a+b*sqrt(2)+c*i+d*i*sqrt(2).
AI = tuple[int, int, int, int]
ZERO: AI = (0, 0, 0, 0)
ONE: AI = (1, 0, 0, 0)
MINUS_ONE: AI = (-1, 0, 0, 0)
I: AI = (0, 0, 1, 0)
SQRT2: AI = (0, 1, 0, 0)


def ai_add(x: AI, y: AI) -> AI:
    return tuple(a + b for a, b in zip(x, y))  # type: ignore[return-value]


def ai_neg(x: AI) -> AI:
    return tuple(-a for a in x)  # type: ignore[return-value]


def ai_sub(x: AI, y: AI) -> AI:
    return ai_add(x, ai_neg(y))


def ai_mul(x: AI, y: AI) -> AI:
    a, b, c, d = x
    e, f, g, h = y
    # (a+b*s)(e+f*s) - (c+d*s)(g+h*s)
    re0 = a * e + 2 * b * f - c * g - 2 * d * h
    re1 = a * f + b * e - c * h - d * g
    # (a+b*s)(g+h*s) + (c+d*s)(e+f*s)
    im0 = a * g + 2 * b * h + c * e + 2 * d * f
    im1 = a * h + b * g + c * f + d * e
    return (re0, re1, im0, im1)


def ai_scale(n: int, x: AI) -> AI:
    return tuple(n * a for a in x)  # type: ignore[return-value]


def ai_pow(x: AI, n: int) -> AI:
    out = ONE
    for _ in range(n):
        out = ai_mul(out, x)
    return out


MU4: tuple[AI, ...] = (ONE, I, MINUS_ONE, ai_neg(I))


def ai_times_i_power(x: AI, exponent: int) -> AI:
    exponent %= 4
    if exponent == 0:
        return x
    if exponent == 1:
        a, b, c, d = x
        return (-c, -d, a, b)
    if exponent == 2:
        return ai_neg(x)
    a, b, c, d = x
    return (c, d, -a, -b)


# Elements of Q(sqrt(2),i), used only for projective normalization and the
# complete diagonal-orbit invariant.
F = tuple[Fraction, Fraction, Fraction, Fraction]
FZERO: F = (Fraction(0),) * 4


def f_from_ai(x: AI) -> F:
    return tuple(Fraction(a) for a in x)  # type: ignore[return-value]


def f_add(x: F, y: F) -> F:
    return tuple(a + b for a, b in zip(x, y))  # type: ignore[return-value]


def f_neg(x: F) -> F:
    return tuple(-a for a in x)  # type: ignore[return-value]


def f_mul(x: F, y: F) -> F:
    a, b, c, d = x
    e, f, g, h = y
    return (
        a * e + 2 * b * f - c * g - 2 * d * h,
        a * f + b * e - c * h - d * g,
        a * g + 2 * b * h + c * e + 2 * d * f,
        a * h + b * g + c * f + d * e,
    )


def sqrt_pair_mul(x: tuple[Fraction, Fraction],
                  y: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    a, b = x
    c, d = y
    return (a * c + 2 * b * d, a * d + b * c)


def sqrt_pair_inv(x: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    a, b = x
    denominator = a * a - 2 * b * b
    assert denominator != 0
    return (a / denominator, -b / denominator)


def f_inv(x: F) -> F:
    a, b, c, d = x
    norm = (
        a * a + 2 * b * b + c * c + 2 * d * d,
        2 * a * b + 2 * c * d,
    )
    inv_norm = sqrt_pair_inv(norm)
    real = sqrt_pair_mul((a, b), inv_norm)
    imag = sqrt_pair_mul((-c, -d), inv_norm)
    out = (real[0], real[1], imag[0], imag[1])
    one = f_mul(x, out)
    assert one == (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    return out


def f_div(x: F, y: F) -> F:
    return f_mul(x, f_inv(y))


def projective_key(vector: tuple[AI, ...]) -> tuple[F, ...]:
    first = next(x for x in vector if x != ZERO)
    inverse = f_inv(f_from_ai(first))
    return tuple(f_mul(f_from_ai(x), inverse) for x in vector)


def bits(word: int) -> tuple[int, int, int, int]:
    return tuple((word >> (3 - j)) & 1 for j in range(4))  # type: ignore[return-value]


WORDS = tuple(bits(word) for word in range(16))
EVEN = tuple(word for word, x in enumerate(WORDS) if sum(x) % 2 == 0)
ODD = tuple(word for word, x in enumerate(WORDS) if sum(x) % 2 == 1)
CENTRAL = (0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100)
PAIRS = tuple(combinations(range(4), 2))


def vector_from_positions(positions: tuple[int, ...]) -> int:
    value = 0
    for position in positions:
        value |= 1 << (3 - position)
    return value


def span(generators: tuple[int, ...]) -> frozenset[int]:
    values = {0}
    for generator in generators:
        values |= {x ^ generator for x in tuple(values)}
    return frozenset(values)


def all_linear_subspaces() -> tuple[frozenset[int], ...]:
    spaces = {span(tuple(gens))
              for size in range(5)
              for gens in combinations(range(1, 16), size)}
    assert len(spaces) == 67
    return tuple(sorted(spaces, key=lambda s: (len(s), tuple(sorted(s)))))


def canonical_basis(space: frozenset[int]) -> tuple[int, ...]:
    dimension = (len(space) - 1).bit_length()
    if dimension == 0:
        return ()
    for candidate in combinations(sorted(space - {0}), dimension):
        if span(tuple(candidate)) == space:
            return tuple(candidate)
    raise AssertionError("subspace has no basis")


Signature = tuple[int, ...]  # -1 denotes zero; 0,1,2,3 denote powers of i.


def generate_affine_signatures() -> tuple[tuple[Signature, ...], Counter[int], Counter[int]]:
    signatures: set[Signature] = set()
    support_counts: Counter[int] = Counter()
    phase_counts: Counter[int] = Counter()

    for space in all_linear_subspaces():
        dimension = (len(space) - 1).bit_length()
        basis = canonical_basis(space)
        cosets = {frozenset(origin ^ x for x in space) for origin in range(16)}
        support_counts[dimension] += len(cosets)
        for support in sorted(cosets, key=lambda s: tuple(sorted(s))):
            origin = min(support)
            coordinate_to_word: dict[tuple[int, ...], int] = {}
            for coordinate in product((0, 1), repeat=dimension):
                word = origin
                for coefficient, generator in zip(coordinate, basis):
                    if coefficient:
                        word ^= generator
                coordinate_to_word[coordinate] = word
            assert frozenset(coordinate_to_word.values()) == support

            pair_indices = tuple(combinations(range(dimension), 2))
            local: set[Signature] = set()
            for linear in product(range(4), repeat=dimension):
                for quadratic in product((0, 1), repeat=len(pair_indices)):
                    table = [-1] * 16
                    for coordinate, word in coordinate_to_word.items():
                        exponent = sum(a * x for a, x in zip(linear, coordinate))
                        exponent += 2 * sum(
                            q * coordinate[j] * coordinate[k]
                            for q, (j, k) in zip(quadratic, pair_indices)
                        )
                        table[word] = exponent % 4
                    local.add(tuple(table))
            expected = 4 ** dimension * 2 ** comb(dimension, 2)
            assert len(local) == expected
            phase_counts[dimension] += len(local)
            signatures.update(local)

    expected_supports = Counter({0: 16, 1: 120, 2: 140, 3: 30, 4: 1})
    expected_phases = Counter({0: 16, 1: 480, 2: 4480, 3: 15360, 4: 16384})
    assert support_counts == expected_supports
    assert phase_counts == expected_phases
    assert len(signatures) == 36_720
    ordered = tuple(sorted(signatures))
    return ordered, support_counts, phase_counts


Matrix = tuple[tuple[AI, AI], tuple[AI, AI]]


def matrix_cores() -> tuple[tuple[str, Matrix], ...]:
    cores: list[tuple[str, Matrix]] = [
        ("I", ((ONE, ZERO), (ZERO, ONE))),
        ("X", ((ZERO, ONE), (ONE, ZERO))),
    ]
    sqrt2_mu8_odd = {
        1: ai_add(ONE, I),
        3: ai_add(MINUS_ONE, I),
        5: ai_neg(ai_add(ONE, I)),
        7: ai_add(ONE, ai_neg(I)),
    }
    for exponent in range(8):
        if exponent % 2 == 0:
            r = MU4[exponent // 2]
            core = ((ONE, r), (ONE, ai_neg(r)))
        else:
            # sqrt(2) times the displayed core; the scalar is projective.
            sr = sqrt2_mu8_odd[exponent]
            core = ((SQRT2, sr), (SQRT2, ai_neg(sr)))
        cores.append((f"K_zeta8^{exponent}", core))

    one_plus_sqrt2 = ai_add(ONE, SQRT2)
    one_minus_sqrt2 = ai_sub(ONE, SQRT2)
    for u_exponent in range(4):
        u = MU4[u_exponent]
        for epsilon in (1, -1):
            plus = one_plus_sqrt2 if epsilon == 1 else one_minus_sqrt2
            minus = one_minus_sqrt2 if epsilon == 1 else one_plus_sqrt2
            core = ((ONE, ai_mul(u, plus)), (ONE, ai_mul(u, minus)))
            cores.append((f"L_i^{u_exponent}_{epsilon:+d}", core))

    assert len(cores) == 18
    assert len({projective_key(tuple(entry for row in matrix for entry in row))
                for _, matrix in cores}) == 18
    for _, matrix in cores:
        determinant = ai_sub(ai_mul(matrix[0][0], matrix[1][1]),
                             ai_mul(matrix[0][1], matrix[1][0]))
        assert determinant != ZERO
    return tuple(cores)


def transformed_edge(matrix: Matrix) -> tuple[AI, AI, AI, AI]:
    # K^T X K.
    out: list[AI] = []
    for a in range(2):
        for b in range(2):
            out.append(ai_add(ai_mul(matrix[0][a], matrix[1][b]),
                              ai_mul(matrix[1][a], matrix[0][b])))
    return tuple(out)  # type: ignore[return-value]


def check_edge_cores(cores: tuple[tuple[str, Matrix], ...]) -> None:
    expected: set[tuple[F, ...]] = set()
    expected.add(projective_key((ZERO, ONE, ONE, ZERO)))
    for eta in MU4:
        expected.add(projective_key((ONE, ZERO, ZERO, eta)))
    for u in MU4:
        expected.add(projective_key((ONE, u, u, ai_neg(ai_mul(u, u)))))
    histogram = Counter(projective_key(transformed_edge(matrix))
                        for _, matrix in cores)
    assert set(histogram) == expected
    assert Counter(histogram.values()) == Counter({2: 9})


def signature_support(signature: Signature) -> tuple[int, ...]:
    return tuple(index for index, exponent in enumerate(signature) if exponent >= 0)


def signature_values(signature: Signature) -> tuple[AI, ...]:
    return tuple(ZERO if exponent < 0 else MU4[exponent]
                 for exponent in signature)


def diagonal_invariant(values: tuple[AI, ...]) -> tuple[F, ...]:
    endpoint0 = f_from_ai(values[0])
    endpoint1 = f_from_ai(values[15])
    denominator = f_mul(endpoint0, endpoint1)
    assert denominator != FZERO
    central = tuple(f_from_ai(values[index]) for index in CENTRAL)
    return tuple(f_div(f_mul(central[i], central[j]), denominator)
                 for i in range(6) for j in range(i, 6))


def tensor_coefficients(matrix: Matrix) -> tuple[tuple[tuple[AI, ...], ...], ...]:
    rows: list[tuple[tuple[AI, ...], ...]] = []
    for output_word in WORDS:
        row: list[tuple[AI, ...]] = []
        for input_word in WORDS:
            coefficient = ONE
            for output_bit, input_bit in zip(output_word, input_word):
                coefficient = ai_mul(coefficient, matrix[output_bit][input_bit])
            row.append(tuple(ai_times_i_power(coefficient, phase) for phase in range(4)))
        rows.append(tuple(row))
    return tuple(rows)


def dot_signature(coefficients: tuple[tuple[tuple[AI, ...], ...], ...],
                  output: int, supported: tuple[tuple[int, int], ...]) -> AI:
    value = ZERO
    row = coefficients[output]
    for input_word, phase in supported:
        value = ai_add(value, row[input_word][phase])
    return value


def check_affine_core_enumeration(signatures: tuple[Signature, ...],
                                  cores: tuple[tuple[str, Matrix], ...]) -> dict[str, int]:
    raw_references = tuple(
        signature for signature in signatures
        if all(signature[index] < 0 for index in ODD)
        and signature[0] >= 0 and signature[15] >= 0
    )
    raw_histogram = Counter(len(signature_support(signature))
                            for signature in raw_references)
    assert raw_histogram == Counter({2: 4, 4: 96, 8: 512})

    invariant_references: defaultdict[tuple[F, ...], list[Signature]] = defaultdict(list)
    invariant_support: dict[tuple[F, ...], int] = {}
    for signature in raw_references:
        invariant = diagonal_invariant(signature_values(signature))
        invariant_references[invariant].append(signature)
        invariant_support.setdefault(invariant, len(signature_support(signature)))
        assert invariant_support[invariant] == len(signature_support(signature))
    orbit_histogram = Counter(invariant_support.values())
    assert orbit_histogram == Counter({2: 1, 4: 24, 8: 256})
    assert len(invariant_references) == 281

    supported_signatures = tuple(
        tuple((index, exponent) for index, exponent in enumerate(signature)
              if exponent >= 0)
        for signature in signatures
    )
    per_core: dict[str, int] = {}
    retained_total = 0
    seen_invariants: set[tuple[F, ...]] = set()
    for label, matrix in cores:
        coefficients = tensor_coefficients(matrix)
        retained = 0
        for supported in supported_signatures:
            if any(dot_signature(coefficients, output, supported) != ZERO
                   for output in ODD):
                continue
            endpoint0 = dot_signature(coefficients, 0, supported)
            endpoint1 = dot_signature(coefficients, 15, supported)
            if endpoint0 == ZERO or endpoint1 == ZERO:
                continue
            values = [ZERO] * 16
            values[0] = endpoint0
            values[15] = endpoint1
            for output in CENTRAL:
                values[output] = dot_signature(coefficients, output, supported)
            support_size = sum(value != ZERO for value in values)
            invariant = diagonal_invariant(tuple(values))
            assert invariant in invariant_references
            assert invariant_support[invariant] == support_size
            seen_invariants.add(invariant)

            # Do not merely look up the invariant: explicitly recover the
            # proportional central layer and verify the endpoint square law.
            reference = signature_values(invariant_references[invariant][0])
            nonzero_central = next((index for index in CENTRAL
                                    if values[index] != ZERO), None)
            if nonzero_central is not None:
                lam = f_div(
                    f_mul(f_from_ai(values[nonzero_central]), f_from_ai(reference[0])),
                    f_mul(f_from_ai(reference[nonzero_central]), f_from_ai(values[0])),
                )
                for index in CENTRAL:
                    left = f_mul(f_from_ai(values[index]), f_from_ai(reference[0]))
                    right = f_mul(lam, f_mul(f_from_ai(reference[index]),
                                             f_from_ai(values[0])))
                    assert left == right
                normalized_endpoint = f_div(f_from_ai(values[15]), f_from_ai(values[0]))
                reference_endpoint = f_div(f_from_ai(reference[15]), f_from_ai(reference[0]))
                assert normalized_endpoint == f_mul(f_mul(lam, lam), reference_endpoint)
            else:
                assert support_size == 2
            retained += 1
        per_core[label] = retained
        retained_total += retained

    assert retained_total == 4_200
    assert Counter(per_core.values()) == Counter({612: 6, 44: 12})
    assert seen_invariants == set(invariant_references)
    return per_core


def logical_support_four(signature: Signature) -> tuple[int, int]:
    central = tuple(index for index in CENTRAL if signature[index] >= 0)
    assert len(central) == 2 and central[0] ^ central[1] == 0b1111
    return central


def check_support_four(raw_references: tuple[Signature, ...]) -> None:
    positive = 0
    negative = 0
    for signature in raw_references:
        if len(signature_support(signature)) != 4:
            continue
        middle0, middle1 = logical_support_four(signature)
        a, b, c, d = (signature[0], signature[middle0],
                      signature[middle1], signature[15])
        cross_exponent = (a + d - b - c) % 4
        assert cross_exponent in (0, 2)
        if cross_exponent == 0:
            positive += 1
            assert ai_mul(MU4[a], MU4[d]) == ai_mul(MU4[b], MU4[c])
        else:
            negative += 1
            # The displayed two-copy logical contraction.
            output = (
                ai_scale(2, ai_mul(MU4[a], MU4[b])),
                ai_add(ai_mul(MU4[a], MU4[d]), ai_mul(MU4[b], MU4[c])),
                ai_add(ai_mul(MU4[a], MU4[d]), ai_mul(MU4[b], MU4[c])),
                ai_scale(2, ai_mul(MU4[c], MU4[d])),
            )
            assert output[0] != ZERO and output[3] != ZERO
            assert output[1] == output[2] == ZERO
    assert (positive, negative) == (48, 48)


def self_loop(values: tuple[AI, ...], i: int, j: int) -> tuple[AI, ...]:
    remaining = tuple(k for k in range(4) if k not in (i, j))
    output: list[AI] = []
    for external in product((0, 1), repeat=2):
        coefficient = ZERO
        for bit_i in (0, 1):
            word = [0] * 4
            word[i] = bit_i
            word[j] = 1 - bit_i
            for port, bit in zip(remaining, external):
                word[port] = bit
            coefficient = ai_add(coefficient, values[vector_from_positions(
                tuple(port for port, bit in enumerate(word) if bit)
            )])
        output.append(coefficient)
    return tuple(output)


def two_copy(values: tuple[AI, ...], i: int, j: int,
             crossed: bool) -> tuple[AI, ...]:
    remaining = tuple(k for k in range(4) if k not in (i, j))
    output: list[AI] = []
    for external in product((0, 1), repeat=4):
        coefficient = ZERO
        for bit_i, bit_j in product((0, 1), repeat=2):
            first = [0] * 4
            second = [0] * 4
            first[i], first[j] = bit_i, bit_j
            if crossed:
                second[j], second[i] = 1 - bit_i, 1 - bit_j
            else:
                second[i], second[j] = 1 - bit_i, 1 - bit_j
            first[remaining[0]], first[remaining[1]] = external[0], external[1]
            second[remaining[0]], second[remaining[1]] = external[2], external[3]
            first_word = vector_from_positions(tuple(k for k, bit in enumerate(first) if bit))
            second_word = vector_from_positions(tuple(k for k, bit in enumerate(second) if bit))
            coefficient = ai_add(coefficient,
                                 ai_mul(values[first_word], values[second_word]))
        output.append(coefficient)
    return tuple(output)


def is_raw_support_four_product(values: tuple[AI, ...]) -> bool:
    support = tuple(index for index, value in enumerate(values) if value != ZERO)
    if 0 not in support or 15 not in support or len(support) != 4:
        return False
    central = tuple(index for index in support if index not in (0, 15))
    if len(central) != 2 or central[0] ^ central[1] != 15:
        return False
    return (ai_mul(values[0], values[15])
            == ai_mul(values[central[0]], values[central[1]]))


def is_dense_product(values: tuple[AI, ...]) -> bool:
    if any(values[index] != ZERO for index in ODD):
        return False
    if any(values[index] == ZERO for index in EVEN):
        return False
    rho = values[0]
    for epsilon in product((1, -1), repeat=4):
        first_pair = vector_from_positions((0, 1))
        t = ai_scale(epsilon[0] * epsilon[1], values[first_pair])
        if any(values[vector_from_positions((i, j))]
               != ai_scale(epsilon[i] * epsilon[j], t)
               for i, j in PAIRS):
            continue
        epsilon_product = epsilon[0] * epsilon[1] * epsilon[2] * epsilon[3]
        if ai_mul(values[15], rho) == ai_scale(epsilon_product, ai_mul(t, t)):
            return True
    return False


def mu4_ratio(numerator: AI, denominator: AI) -> int:
    assert numerator != ZERO and denominator != ZERO
    matches = [exponent for exponent, root in enumerate(MU4)
               if numerator == ai_mul(root, denominator)]
    assert len(matches) == 1
    return matches[0]


def check_full_even_partition(raw_references: tuple[Signature, ...]) -> Counter[int]:
    full = tuple(signature for signature in raw_references
                 if len(signature_support(signature)) == 8)
    assert len(full) == 512
    dense: set[Signature] = set()
    rank_one: set[Signature] = set()
    residual: list[Signature] = []
    for signature in full:
        values = signature_values(signature)
        if is_dense_product(values):
            dense.add(signature)
            continue
        has_rank_one = False
        for i, j in PAIRS:
            loop = self_loop(values, i, j)
            assert loop[0] == loop[3] == ZERO
            if (loop[1] == ZERO) != (loop[2] == ZERO):
                has_rank_one = True
                break
        if has_rank_one:
            rank_one.add(signature)
        else:
            residual.append(signature)
    assert (len(dense), len(rank_one), len(residual)) == (16, 336, 160)

    newly_covered = [0, 0, 0]
    ratio_histogram: Counter[int] = Counter()
    for signature in residual:
        values = signature_values(signature)
        chosen: tuple[AI, ...] | None = None
        for stage, j in enumerate((1, 2, 3)):
            output = two_copy(values, 0, j, crossed=True)
            if is_raw_support_four_product(output):
                chosen = output
                newly_covered[stage] += 1
                break
        assert chosen is not None
        central = tuple(index for index in CENTRAL if chosen[index] != ZERO)
        assert len(central) == 2
        # Copy-exchange symmetry gives c E_r tensor E_r.
        assert chosen[central[0]] == chosen[central[1]]
        ratio_histogram[mu4_ratio(chosen[central[0]], chosen[0])] += 1
    assert newly_covered == [96, 48, 16]
    assert ratio_histogram == Counter({0: 56, 2: 56, 1: 24, 3: 24})
    return ratio_histogram


def gaussian_norm(x: AI) -> int:
    a, b, c, d = x
    assert b == d == 0
    return a * a + c * c


OddState = tuple[int, int, int, int, int, int, int, int]


def all_odd_states() -> tuple[OddState, ...]:
    return tuple((0, alpha1, alpha2, alpha3, beta0, beta1, beta2, beta3)
                 for alpha1, alpha2, alpha3, beta0, beta1, beta2, beta3
                 in product(range(4), repeat=7))


def odd_values(state: OddState) -> tuple[AI, ...]:
    values = [ZERO] * 16
    a = state[:4]
    b = state[4:]
    for j in range(4):
        values[vector_from_positions((j,))] = MU4[a[j]]
        values[15 ^ vector_from_positions((j,))] = MU4[b[j]]
    return tuple(values)


def twelve_loop_pairs(state: OddState) -> tuple[tuple[AI, AI], ...]:
    a = tuple(MU4[exponent] for exponent in state[:4])
    b = tuple(MU4[exponent] for exponent in state[4:])
    loops: list[tuple[AI, AI]] = []
    for i, j in PAIRS:
        loops.append((ai_add(a[i], a[j]), ai_add(b[i], b[j])))
    for i, j in PAIRS:
        k, l = tuple(port for port in range(4) if port not in (i, j))
        loops.append((ai_add(a[l], b[k]), ai_add(a[k], b[l])))
    assert len(loops) == 12
    return tuple(loops)


def loop_class(state: OddState) -> str:
    loops = twelve_loop_pairs(state)
    if any((left == ZERO) != (right == ZERO) for left, right in loops):
        return "rank-one"
    if any(left != ZERO and right != ZERO
           and gaussian_norm(left) != gaussian_norm(right)
           for left, right in loops):
        return "infinite-order"
    return "stalled"


def is_exceptional64(state: OddState) -> bool:
    differences = tuple((state[4 + j] - state[j]) % 4 for j in range(4))
    return (differences in ((0, 0, 0, 0), (2, 2, 2, 2))
            and sum(state[:4]) % 4 in (1, 3))


def odd_exponent_at(state: OddState, word: tuple[int, int, int, int],
                    delta: int = 0) -> int:
    weight = sum(word)
    assert weight in (1, 3)
    if weight == 1:
        return state[word.index(1)]
    return (state[4 + word.index(0)] + delta) % 4


def affine_delta_test(state: OddState, delta: int) -> bool:
    phase: dict[tuple[int, int, int], int] = {}
    for y in product((0, 1), repeat=3):
        word = (y[0], y[1], y[2], 1 ^ y[0] ^ y[1] ^ y[2])
        phase[y] = odd_exponent_at(state, word, delta)
    base = phase[(0, 0, 0)]
    phase = {y: (value - base) % 4 for y, value in phase.items()}
    unit = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    linear = tuple(phase[y] for y in unit)
    quadratic: dict[tuple[int, int], int] = {}
    for j, k in combinations(range(3), 2):
        y = tuple(1 if index in (j, k) else 0 for index in range(3))
        quadratic[(j, k)] = (phase[y] - linear[j] - linear[k]) % 4
        if quadratic[(j, k)] not in (0, 2):
            return False
    predicted = (sum(linear) + sum(quadratic.values())) % 4
    return phase[(1, 1, 1)] == predicted


def shifted_odd_state(state: OddState, delta: int) -> OddState:
    return tuple(state[:4]) + tuple((x + delta) % 4 for x in state[4:])  # type: ignore[return-value]


def affine_odd_family() -> dict[OddState, tuple[int, int, int, int, int, int]]:
    family: dict[OddState, tuple[int, int, int, int, int, int]] = {}
    for alpha, beta, gamma in product(range(4), repeat=3):
        for x, y, z in product((1, -1), repeat=3):
            sx = 0 if x == 1 else 2
            sy = 0 if y == 1 else 2
            sz = 0 if z == 1 else 2
            state: OddState = (
                0, alpha, beta, gamma,
                (sx + sy + sz + alpha + beta + gamma) % 4,
                (sz + beta + gamma) % 4,
                (sy + alpha + gamma) % 4,
                (sx + alpha + beta) % 4,
            )
            assert state not in family
            family[state] = (alpha, beta, gamma, x, y, z)
    assert len(family) == 512
    return family


def has_rank_one_x_loop(state: OddState) -> bool:
    a = tuple(MU4[exponent] for exponent in state[:4])
    b = tuple(MU4[exponent] for exponent in state[4:])
    return any((ai_add(a[i], a[j]) == ZERO)
               != (ai_add(b[i], b[j]) == ZERO)
               for i, j in PAIRS)


def check_affine_odd_family() -> set[OddState]:
    family = affine_odd_family()
    rank_one = {state for state in family if has_rank_one_x_loop(state)}
    no_rank = set(family) - rank_one
    assert (len(rank_one), len(no_rank)) == (336, 176)

    locus_a: set[OddState] = set()
    locus_b: set[OddState] = set()
    for state, (alpha, beta, gamma, x, y, z) in family.items():
        if x == y == z:
            locus_a.add(state)
        if x != y or y != z:
            squares = ((2 * alpha) % 4, (2 * beta) % 4, (2 * gamma) % 4)
            target = (0 if x * y == 1 else 2,
                      0 if x * z == 1 else 2,
                      0 if y * z == 1 else 2)
            if squares == target:
                locus_b.add(state)
    assert len(locus_a) == 128
    assert len(locus_b) == 48
    assert locus_a.isdisjoint(locus_b)
    assert no_rank == locus_a | locus_b

    covered: set[OddState] = set()
    for state, (alpha, beta, gamma, x, y, z) in family.items():
        values = odd_values(state)
        predicates = (
            x == y and (2 * beta) % 4 == (2 * gamma) % 4,
            x == z and (2 * alpha) % 4 == (2 * gamma) % 4,
            y == z and (2 * alpha) % 4 == (2 * beta) % 4,
        )
        actual = tuple(is_dense_product(two_copy(values, 0, j, crossed=False))
                       for j in (1, 2, 3))
        assert actual == predicates
        if state in no_rank:
            assert any(actual)
            covered.add(state)
    assert covered == no_rank
    return no_rank


def check_exceptional_gadget(state: OddState) -> None:
    values = odd_values(state)
    output = two_copy(values, 0, 1, crossed=False)
    a = tuple(MU4[exponent] for exponent in state[:4])
    b = tuple(MU4[exponent] for exponent in state[4:])
    expected = [ZERO] * 16
    expected[0] = ai_scale(2, ai_mul(a[0], a[1]))
    expected[15] = ai_scale(2, ai_mul(b[0], b[1]))
    expected[0b0011] = expected[0b1100] = ai_add(ai_mul(a[0], b[0]),
                                                          ai_mul(a[1], b[1]))
    expected[0b0101] = ai_scale(2, ai_mul(a[3], b[2]))
    expected[0b1010] = ai_scale(2, ai_mul(a[2], b[3]))
    expected[0b0110] = expected[0b1001] = ai_add(ai_mul(a[2], b[2]),
                                                          ai_mul(a[3], b[3]))
    assert output == tuple(expected)
    square_sums = (ai_add(ai_mul(a[0], a[0]), ai_mul(a[1], a[1])),
                   ai_add(ai_mul(a[2], a[2]), ai_mul(a[3], a[3])))
    assert sum(value == ZERO for value in square_sums) == 1
    support = tuple(index for index, value in enumerate(output) if value != ZERO)
    assert len(support) == 6
    assert 0 in support and 15 in support
    assert all(index in EVEN for index in support)


def check_mu4_residue() -> None:
    states = all_odd_states()
    assert len(states) == 16_384
    classes = Counter(loop_class(state) for state in states)
    assert classes == Counter({"rank-one": 15_792,
                               "infinite-order": 224,
                               "stalled": 368})
    stalled = {state for state in states if loop_class(state) == "stalled"}
    exceptional = {state for state in stalled if is_exceptional64(state)}
    assert len(exceptional) == 64
    assert {state for state in states if is_exceptional64(state)} == exceptional
    for state in exceptional:
        check_exceptional_gadget(state)

    affine_family = set(affine_odd_family())
    delta_histogram: Counter[tuple[int, ...]] = Counter()
    for state in stalled - exceptional:
        valid = tuple(delta for delta in range(4) if affine_delta_test(state, delta))
        assert valid in ((0, 2), (1, 3))
        assert all(shifted_odd_state(state, delta) in affine_family for delta in valid)
        assert all(shifted_odd_state(state, delta) not in affine_family
                   for delta in set(range(4)) - set(valid))
        delta_histogram[valid] += 1
    assert delta_histogram == Counter({(0, 2): 176, (1, 3): 128})


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    signatures, support_counts, phase_counts = generate_affine_signatures()
    cores = matrix_cores()
    check_edge_cores(cores)

    raw_references = tuple(
        signature for signature in signatures
        if all(signature[index] < 0 for index in ODD)
        and signature[0] >= 0 and signature[15] >= 0
    )
    per_core = check_affine_core_enumeration(signatures, cores)
    check_support_four(raw_references)
    ratio_histogram = check_full_even_partition(raw_references)
    no_rank_odd = check_affine_odd_family()
    check_mu4_residue()

    print("affine supports by dimension:",
          [support_counts[d] for d in range(5)])
    print("affine signatures by support dimension:",
          [phase_counts[d] for d in range(5)])
    print("basis cores / retained pairs:", len(cores), sum(per_core.values()))
    print("normalized reference states / diagonal orbits: 612 / 281")
    print("full-even partition: 16 + 336 + 96 + 48 + 16 = 512")
    print("extracted r histogram (1,-1,i,-i):",
          [ratio_histogram[k] for k in (0, 2, 1, 3)])
    print("full-odd affine no-rank residue:", len(no_rank_odd))
    print("twelve-loop partition: 15792 + 224 + 368 = 16384")
    print("stalled-state split: 64 exceptional support-six exits + "
          "304 nonexceptional states (two valid affine phase shifts each)")
    print("AFFINE ENDPOINT-NONDEGENERATE Q4 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
