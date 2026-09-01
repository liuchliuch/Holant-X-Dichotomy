#!/usr/bin/env python3
"""Exact replay of the second distinguished-involution S4 q4 terminals.

The analytic transport uses Q(sqrt(2),i), represented by four rational
coefficients.  The similarity H is used only to reconstruct the stated
second-form coefficient tables.  Every kernel and every local dressing
which is asserted to be physical is separately constructed as H^{-1} U H
with U in the first octahedral group.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import verify_a4_s4_certificates as base


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )


@dataclass(frozen=True, slots=True)
class F:
    """(a+b sqrt(2)) + i(c+d sqrt(2)), a,b,c,d in Q."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)
    c: Fraction = Fraction(0)
    d: Fraction = Fraction(0)

    @staticmethod
    def rational(value: int | Fraction) -> "F":
        return F(value if isinstance(value, Fraction) else Fraction(value))

    @staticmethod
    def coerce(value: "F | int | Fraction") -> "F":
        return value if isinstance(value, F) else F.rational(value)

    def __add__(self, other: "F | int | Fraction") -> "F":
        o = F.coerce(other)
        return F(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self) -> "F":
        return F(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, other: "F | int | Fraction") -> "F":
        return self + (-F.coerce(other))

    def __rsub__(self, other: "F | int | Fraction") -> "F":
        return F.coerce(other) - self

    def __mul__(self, other: "F | int | Fraction") -> "F":
        o = F.coerce(other)

        def rmul(x0: Fraction, x1: Fraction,
                 y0: Fraction, y1: Fraction) -> tuple[Fraction, Fraction]:
            return x0 * y0 + 2 * x1 * y1, x0 * y1 + x1 * y0

        rr = rmul(self.a, self.b, o.a, o.b)
        ii = rmul(self.c, self.d, o.c, o.d)
        ri = rmul(self.a, self.b, o.c, o.d)
        ir = rmul(self.c, self.d, o.a, o.b)
        return F(rr[0] - ii[0], rr[1] - ii[1],
                 ri[0] + ir[0], ri[1] + ir[1])

    __rmul__ = __mul__

    def inverse(self) -> "F":
        def rmul(x0: Fraction, x1: Fraction,
                 y0: Fraction, y1: Fraction) -> tuple[Fraction, Fraction]:
            return x0 * y0 + 2 * x1 * y1, x0 * y1 + x1 * y0

        xx = rmul(self.a, self.b, self.a, self.b)
        yy = rmul(self.c, self.d, self.c, self.d)
        p, q = xx[0] + yy[0], xx[1] + yy[1]
        norm = p * p - 2 * q * q
        if norm == 0:
            raise ZeroDivisionError
        n0, n1 = p / norm, -q / norm
        real = rmul(self.a, self.b, n0, n1)
        imag = rmul(-self.c, -self.d, n0, n1)
        return F(real[0], real[1], imag[0], imag[1])

    def __truediv__(self, other: "F | int | Fraction") -> "F":
        return self * F.coerce(other).inverse()

    def is_zero(self) -> bool:
        return self == Z


Z = F.rational(0)
O = F.rational(1)
T = F.rational(2)
II = F(Fraction(0), Fraction(0), Fraction(1), Fraction(0))
S = F(Fraction(0), Fraction(1), Fraction(0), Fraction(0))
Matrix = tuple[F, F, F, F]
Vector = tuple[F, ...]


def from_e(value: base.E) -> F:
    assert value.b == value.d == 0
    return F(value.a, Fraction(0), value.c, Fraction(0))


def from_e_vector(vector: Sequence[base.E]) -> Vector:
    return tuple(from_e(value) for value in vector)


def fm(entries: Sequence[F | int | Fraction]) -> Matrix:
    assert len(entries) == 4
    return tuple(F.coerce(value) for value in entries)  # type: ignore[return-value]


def mm(left: Matrix, right: Matrix) -> Matrix:
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def det(matrix: Matrix) -> F:
    return matrix[0] * matrix[3] - matrix[1] * matrix[2]


def inv(matrix: Matrix) -> Matrix:
    determinant = det(matrix)
    return tuple(value / determinant
                 for value in (matrix[3], -matrix[1], -matrix[2], matrix[0]))  # type: ignore[return-value]


def entry(matrix: Matrix, row: int, column: int) -> F:
    return matrix[2 * row + column]


Q0 = from_e_vector(base.Q0)
Q1 = from_e_vector(base.Q1)
Q2 = from_e_vector(base.Q2)
Q3 = from_e_vector(base.Q3)
X = fm((0, 1, 1, 0))
IDENTITY = fm((1, 0, 0, 1))
H = fm((S, -II * S, O - II, O + II))
HINV = inv(H)


def add4(*matrices: Matrix) -> Matrix:
    return tuple(sum((matrix[index] for matrix in matrices), Z)
                 for index in range(4))  # type: ignore[return-value]


def signed_oct(e1: int, e2: int, e3: int) -> Matrix:
    return tuple(Q0[index] + e1 * Q1[index]
                 + e2 * Q2[index] + e3 * Q3[index]
                 for index in range(4))  # type: ignore[return-value]


def theta(matrix: Matrix) -> Matrix:
    return mm(mm(HINV, matrix), H)


def projective(vector: Sequence[F]) -> Vector:
    pivot = next(value for value in vector if not value.is_zero())
    return tuple(value / pivot for value in vector)


def local_transform(vector: Sequence[F], matrices: Sequence[Matrix]) -> Vector:
    output = []
    for out_bits in itertools.product((0, 1), repeat=4):
        total = Z
        for in_bits in itertools.product((0, 1), repeat=4):
            index = 8 * in_bits[0] + 4 * in_bits[1] + 2 * in_bits[2] + in_bits[3]
            coefficient = vector[index]
            for port in range(4):
                coefficient *= entry(matrices[port], out_bits[port], in_bits[port])
            total += coefficient
        output.append(total)
    return tuple(output)


def contract(vector: Sequence[F], pair: tuple[int, int], kernel: Matrix) -> Matrix:
    remaining = tuple(port for port in range(4) if port not in pair)
    output = []
    for residual in itertools.product((0, 1), repeat=2):
        total = Z
        for internal in itertools.product((0, 1), repeat=2):
            bits = [0] * 4
            bits[pair[0]], bits[pair[1]] = internal
            bits[remaining[0]], bits[remaining[1]] = residual
            index = 8 * bits[0] + 4 * bits[1] + 2 * bits[2] + bits[3]
            total += vector[index] * entry(kernel, internal[0], internal[1])
        output.append(total)
    return tuple(output)  # type: ignore[return-value]


def quaternion_coordinates(matrix: Matrix) -> Vector:
    a, b, c, d = matrix
    return ((a + d) / T, (b + c) / (T * II),
            (b - c) / T, (a - d) / (T * II))


def even_vector(vector: Sequence[F]) -> Vector:
    return tuple(vector[index] for index in (0, 3, 5, 6, 9, 10, 12, 15))


def all_odd_zero(vector: Sequence[F]) -> bool:
    return all(vector[index].is_zero() for index in range(16)
               if sum(base.bits4(index)) % 2 == 1)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    base.build_group("S4")
    a, b, c, named = base.named_bridge_vectors()
    bridge = [base.zero_vector(16), base.vscale(Fraction(1, 2), a),
              base.vscale(Fraction(1, 2), b), *named.values()]
    inverse_name = {value: name for name, value in named.items()}
    target = base.vscale(Fraction(1, 2), base.vadd(a, b))
    triples = base.sum_equation_solutions(bridge, target)

    separator_kernels = {
        "R13++": (1, 1, 1),
        "R13--": (1, -1, 1),
        "R03+-": (1, 1, -1),
        "R03-+": (1, 1, 1),
        "R01+-": (1, 1, -1),
        "R01-+": (1, 1, 1),
    }
    expected_transfers = {
        "R13++": (2, 1, 1, 0),
        "R13--": (0, 1, 1, 2),
        "R03+-": (1, 2, 0, 1),
        "R03-+": (1, 0, 2, 1),
        "R01+-": (2, 1, -1, 0),
        "R01-+": (2, -1, 1, 0),
    }
    raw_dressings = {
        "R0": ([theta(signed_oct(1, 1, -1))] * 4,
               (1, 1, -1, 1, 1, -1, 1, 1)),
        "R1": ([IDENTITY] * 4,
               (1, -1, 1, 1, 1, 1, -1, 1)),
        "R3": ([theta(signed_oct(1, 1, 1))] * 4,
               (1, -1, 1, 1, 1, 1, -1, 1)),
    }

    separator_count = raw_count = deficient_count = 0
    deficient_transfers = [
        (3, -1, 1, 1), (1, -1, 0, 1), (3, -1, 1, 1),
        (1, -1, 0, 1), (3, -1, -1, 1), (3, -1, -1, 1),
    ]
    deficient_index = 0
    for triple in triples:
        columns = (base.vscale(Fraction(1, 2), c), *triple)
        rank = base.card_map_rank(columns)
        if rank == 2:
            tensor = base.tensor6_from_columns(columns)
            q = base.contract_tensor6_pair(tensor, (0, 2), base.Q0)
            qprime = local_transform(from_e_vector(q), [HINV] * 4)
            kernel = theta(signed_oct(-1, -1, -1))
            binary = contract(qprime, (0, 1), kernel)
            assert det(binary) == F.rational(Fraction(-3, 512))
            pulled = mm(mm(H, mm(binary, X)), HINV)
            assert projective(quaternion_coordinates(pulled)) == projective(
                tuple(F.rational(value) for value in deficient_transfers[deficient_index])
            )
            deficient_index += 1
            deficient_count += 1
            continue
        if rank != 4:
            continue

        names = tuple(inverse_name[value] for value in triple)
        q = base.map_value(columns, (base.ONE, base.ONE, base.ZERO, base.ZERO))
        qprime = local_transform(from_e_vector(q), [HINV] * 4)
        first = names[0]
        if first in separator_kernels:
            kernel = theta(signed_oct(*separator_kernels[first]))
            binary = contract(qprime, (0, 1), kernel)
            assert det(binary) == F.rational(Fraction(-3, 256))
            pulled = mm(mm(H, mm(binary, X)), HINV)
            expected = tuple(F.rational(value) for value in expected_transfers[first])
            assert projective(quaternion_coordinates(pulled)) == projective(expected)
            separator_count += 1
        else:
            matrices, expected_entries = raw_dressings[first]
            raw = local_transform(qprime, matrices)
            assert all_odd_zero(raw)
            expected = tuple(F.rational(value) for value in expected_entries)
            assert projective(even_vector(raw)) == projective(expected)
            assert not raw[0].is_zero() and not raw[15].is_zero()
            raw_count += 1

    assert (deficient_count, separator_count, raw_count) == (6, 12, 12)
    print({
        "deficient_uniform_outsiders": deficient_count,
        "separator_outsiders": separator_count,
        "actual_dressed_outputs": raw_count,
        "field": "Q(sqrt(2),i)",
    })
    print("S4 SECOND-FORM EXACT TERMINAL CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
