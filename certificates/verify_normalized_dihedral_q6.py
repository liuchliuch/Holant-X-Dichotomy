#!/usr/bin/env python3
"""Exact support atlas for normalized-dihedral six-to-four lowering."""

from __future__ import annotations

import argparse
from itertools import combinations, product

if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


PORTS = tuple(range(6))
RESIDUAL_PORTS = (2, 3, 4, 5)


def perfect_matchings(ports: tuple[int, ...]):
    """Generate every labelled perfect matching of an even port tuple."""
    if not ports:
        yield ()
        return
    first = ports[0]
    for index in range(1, len(ports)):
        second = ports[index]
        remainder = ports[1:index] + ports[index + 1 :]
        for tail in perfect_matchings(remainder):
            yield ((first, second),) + tail


RESIDUAL_MATCHINGS = tuple(perfect_matchings(RESIDUAL_PORTS))


def endpoint_slice(
    pair_bits: tuple[int, int],
    matching: tuple[tuple[int, int], tuple[int, int]],
    fixed_index: int,
    fixed_parity: int,
    variable_parity: int,
    endpoint_orientation: int,
) -> set[tuple[int, ...]]:
    """One flat endpoint: one monomial edge and one coordinate endpoint."""
    fixed_edge = matching[fixed_index]
    variable_edge = matching[1 - fixed_index]
    words: set[tuple[int, ...]] = set()
    for logical_bit in (0, 1):
        word = [0] * 6
        word[0], word[1] = pair_bits
        word[fixed_edge[0]] = logical_bit
        word[fixed_edge[1]] = logical_bit ^ fixed_parity
        word[variable_edge[0]] = endpoint_orientation
        word[variable_edge[1]] = endpoint_orientation ^ variable_parity
        words.add(tuple(word))
    return words


def four_flat_support(
    even_matching: tuple[tuple[int, int], tuple[int, int]],
    even_data: tuple[int, int, int, int],
    odd_matching: tuple[tuple[int, int], tuple[int, int]],
    odd_data: tuple[int, int, int, int],
) -> set[tuple[int, ...]]:
    """The four complementary flat endpoint slices of a six-port pencil."""
    support = endpoint_slice((0, 0), even_matching, *even_data)
    support |= endpoint_slice(
        (1, 1),
        even_matching,
        even_data[0],
        even_data[1],
        even_data[2],
        even_data[3] ^ 1,
    )
    support |= endpoint_slice((0, 1), odd_matching, *odd_data)
    support |= endpoint_slice(
        (1, 0),
        odd_matching,
        odd_data[0],
        odd_data[1],
        odd_data[2],
        odd_data[3] ^ 1,
    )
    assert len(support) == 8
    return support


def is_matching_coset(words: set[tuple[int, ...]]) -> bool:
    """Whether words form one coset of a perfect-matching indicator space."""
    if not words:
        return False
    arity = len(next(iter(words)))
    if len(words) != 2 ** (arity // 2):
        return False
    for matching in perfect_matchings(tuple(range(arity))):
        parity_vectors = {
            tuple(word[left] ^ word[right] for left, right in matching)
            for word in words
        }
        if len(parity_vectors) == 1:
            return True
    return False


def best_nonmatching_contraction(
    support: set[tuple[int, ...]],
) -> tuple[tuple[int, int], int, int] | None:
    """Choose an injective I/X contraction with largest nonmatching output."""
    best = None
    for deleted_pair in combinations(PORTS, 2):
        external = tuple(port for port in PORTS if port not in deleted_pair)
        for kernel_parity in (0, 1):
            selected = [
                word
                for word in support
                if word[deleted_pair[0]] ^ word[deleted_pair[1]]
                == kernel_parity
            ]
            projected = {
                tuple(word[port] for port in external) for word in selected
            }
            # Equality means that every external word has a unique lift.
            if len(projected) != len(selected) or not projected:
                continue
            if is_matching_coset(projected):
                continue
            candidate = (deleted_pair, kernel_parity, len(projected))
            if best is None or candidate[2] > best[2]:
                best = candidate
    return best


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    total = 0
    matching_products = 0
    output_histogram: dict[int, int] = {}

    # Per sector: fixed edge, fixed parity, variable parity, endpoint choice.
    sector_data = tuple(product((0, 1), repeat=4))
    for even_matching, odd_matching in product(RESIDUAL_MATCHINGS, repeat=2):
        for even_data, odd_data in product(sector_data, repeat=2):
            total += 1
            support = four_flat_support(
                even_matching, even_data, odd_matching, odd_data
            )
            if is_matching_coset(support):
                matching_products += 1
                continue
            witness = best_nonmatching_contraction(support)
            assert witness is not None
            output_size = witness[2]
            assert output_size in (6, 8)
            output_histogram[output_size] = (
                output_histogram.get(output_size, 0) + 1
            )

    assert total == 3**2 * 2**8 == 2304
    assert matching_products == 96
    assert output_histogram == {6: 1152, 8: 1056}
    assert matching_products + sum(output_histogram.values()) == total

    print(
        "normalized-dihedral q6 atlas: "
        f"{total} patterns = {matching_products} matching products "
        f"+ {output_histogram[6]} support-six lowerings "
        f"+ {output_histogram[8]} support-eight lowerings"
    )
    print("NORMALIZED-DIHEDRAL Q6 SUPPORT LOWERING EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
