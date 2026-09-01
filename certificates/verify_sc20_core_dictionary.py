#!/usr/bin/env python3
"""Verify the exact dictionary between the paper's H6/RM8 and SC20's f6/f8."""

from __future__ import annotations

import argparse
from itertools import product

if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


BITS6 = tuple(product((0, 1), repeat=6))
BITS8 = tuple(product((0, 1), repeat=8))


def h6_value(x: tuple[int, ...]) -> int:
    """Return the H6 coefficient specified in the companion paper."""
    if sum(x) % 2:
        return 0
    x1, x2, x3, _x4, x5, x6 = x
    phase = x1 + x1 * x2 + x1 * x6 + x2 * x5 + x3 * x5 + x3 * x6
    return -1 if phase % 2 else 1


def sc20_hat_f6_value(x: tuple[int, ...]) -> int:
    """Shao--Cai's transformed six-ary signature in their Section 6."""
    if sum(x) % 2:
        return 0
    x1, x2, x3, x4, x5, x6 = x
    phase = x1 * x2 + x2 * x3 + x1 * x3 + x1 * x4 + x2 * x5 + x3 * x6
    return -1 if phase % 2 else 1


def verify_h6() -> None:
    # SC20 ports (1,...,6) receive the paper's ports (1,3,5,4,6,2).
    permutation = (0, 2, 4, 3, 5, 1)
    for x in BITS6:
        y = tuple(x[j] for j in permutation)
        assert h6_value(x) == sc20_hat_f6_value(y)


def sc20_f8_support(x: tuple[int, ...]) -> bool:
    """The four parity equations in SC20, Eq. (8.1)."""
    return (
        sum(x[0:4]) % 2 == 0
        and sum(x[4:8]) % 2 == 0
        and (x[0] + x[1] + x[4] + x[5]) % 2 == 0
        and (x[0] + x[2] + x[4] + x[6]) % 2 == 0
    )


def rm13_support() -> set[tuple[int, ...]]:
    """Truth tables of affine functions on F_2^3, in lexicographic point order."""
    points = tuple(product((0, 1), repeat=3))
    words: set[tuple[int, ...]] = set()
    for c, a, b, d in product((0, 1), repeat=4):
        words.add(tuple(c ^ (a * u) ^ (b * v) ^ (d * w) for u, v, w in points))
    return words


def gaussian_i_power(exponent: int) -> tuple[int, int]:
    """Return i**exponent as an exact Gaussian-integer pair."""
    return ((1, 0), (0, 1), (-1, 0), (0, -1))[exponent % 4]


def k_inverse_numerator(
    output: tuple[int, ...], support: set[tuple[int, ...]]
) -> tuple[int, int]:
    """Numerator of (K^{-1})^{tensor 8} 1[support] before division by 16."""
    real = 0
    imaginary = 0
    for source in support:
        # The numerator of K^{-1} is [[1,-i],[1,i]].  A source zero
        # contributes 1; a source one contributes -i or i according to the
        # output bit.
        exponent = sum((1 if y else 3) for y, x in zip(output, source) if x)
        term_real, term_imaginary = gaussian_i_power(exponent)
        real += term_real
        imaginary += term_imaginary
    return real, imaginary


def verify_rm8() -> None:
    sc20 = {x for x in BITS8 if sc20_f8_support(x)}
    rm13 = rm13_support()
    assert sc20 == rm13
    assert len(sc20) == 16
    assert sorted(sum(x) for x in sc20) == [0] + [4] * 14 + [8]
    for output in BITS8:
        expected = (16, 0) if output in sc20 else (0, 0)
        assert k_inverse_numerator(output, sc20) == expected


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify_h6()
    verify_rm8()
    print("H6(x1,...,x6) = SC20 hat-f6(x1,x3,x5,x4,x6,x2)")
    print("RM8 = SC20 f8 = SC20 hat-f8 in lexicographic F_2^3 coordinate order")
    print("SC20 CORE DICTIONARY: PASS")


if __name__ == "__main__":
    main()
