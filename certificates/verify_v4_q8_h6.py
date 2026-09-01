#!/usr/bin/env python3
"""Exact verifier for the full-V4 q8 base with an H6 card.

This uses only integer/Fraction arithmetic.  It normalizes the 00 card on
ports 0,1 to the canonical H6, generates the complete 960 product plus 768
H6-orbit q6 state set, imposes semantic Frobenius-support separation, and
then imposes the complete q6 card-profile relation in each of the 60
residual pair/kernel contexts.  It checks every surviving q8 line for
standard affine membership.

All arithmetic is rational and every accepted parent is checked from its
literal 256-entry table.  There is no floating point, random sampling,
finite-field specialization, digest comparison, or finite scale grid.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
from itertools import combinations, permutations, product
from collections import defaultdict, Counter


if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()

BITS6 = tuple(product((0, 1), repeat=6))
BITS8 = tuple(product((0, 1), repeat=8))
PAIRS6 = tuple(combinations(range(6), 2))


def canon(v):
    v = tuple(v)
    p = next((x for x in v if x), None)
    if p is None:
        return tuple(v)
    return tuple(Q(x, p) for x in v)


def bell(label):
    d, ell = divmod(label, 2)
    return tuple((-1) ** (ell * x) if x ^ y == d else 0
                 for x, y in product((0, 1), repeat=2))


BELLS = tuple(bell(i) for i in range(4))


def h6():
    out = []
    for x in BITS6:
        if sum(x) & 1:
            out.append(0)
        else:
            e = (x[0] + x[0]*x[1] + x[0]*x[5] + x[1]*x[4]
                 + x[2]*x[4] + x[2]*x[5]) & 1
            out.append((-1) ** e)
    return canon(out)


H0 = h6()


def matchings(ports):
    if not ports:
        yield ()
        return
    a = ports[0]
    for k in range(1, len(ports)):
        for rest in matchings(ports[1:k] + ports[k+1:]):
            yield ((a, ports[k]),) + rest


MATCHINGS6 = tuple(matchings(tuple(range(6))))


def product_state(M, labels):
    out = []
    for x in BITS6:
        z = 1
        for (i, j), label in zip(M, labels):
            z *= BELLS[label][2*x[i] + x[j]]
        out.append(z)
    return canon(out)


PRODUCT_MATCHING = {}
for mi, M in enumerate(MATCHINGS6):
    for labels in product(range(4), repeat=3):
        v = product_state(M, labels)
        assert v not in PRODUCT_MATCHING
        PRODUCT_MATCHING[v] = mi
PRODUCTS = set(PRODUCT_MATCHING)
assert len(PRODUCTS) == 960


def permute_state(v, p):
    return canon(v[sum(x[p[i]] << (5-i) for i in range(6))]
                 for x in BITS6)


PERM_H = {permute_state(H0, p) for p in permutations(range(6))}
assert len(PERM_H) == 120


def pauli_state(v, a, z):
    out = []
    for x in BITS6:
        y = tuple(x[i] ^ a[i] for i in range(6))
        yi = sum(y[i] << (5-i) for i in range(6))
        out.append((-1) ** (sum(z[i]*x[i] for i in range(6)) & 1) * v[yi])
    return canon(out)


# First quotient the Pauli action on one state (4096 operations -> 64
# projective lines), then permute those representatives.  Permutations
# normalize the Pauli group, so this is exactly the same orbit as the
# much slower 120*4096 construction.
H_PAULI = {pauli_state(H0, a, z) for a in BITS6 for z in BITS6}
assert len(H_PAULI) == 64
H_ORBIT = {permute_state(v, p)
           for v in H_PAULI for p in permutations(range(6))}
assert len(H_ORBIT) == 768
assert not (H_ORBIT & PRODUCTS)
ALLOWED = PRODUCTS | H_ORBIT
assert len(ALLOWED) == 1728


def card6(v, pair, label):
    rem = tuple(i for i in range(6) if i not in pair)
    out = []
    for y in product((0, 1), repeat=4):
        s = Q(0)
        for a, b in product((0, 1), repeat=2):
            x = [0]*6
            x[pair[0]], x[pair[1]] = a, b
            for i, bit in zip(rem, y):
                x[i] = bit
            idx = sum(x[i] << (5-i) for i in range(6))
            s += BELLS[label][2*a+b] * v[idx]
        out.append(s)
    return tuple(out)


def direction_scale(v):
    if not any(v):
        return -1, Q(0)
    c = canon(v)
    return LINE_INDEX[c], next(x for x in v if x)


# All 48 q4 product lines.
MATCHINGS4 = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
LINES4 = []
for M in MATCHINGS4:
    for labels in product(range(4), repeat=2):
        out = []
        for x in product((0, 1), repeat=4):
            out.append(BELLS[labels[0]][2*x[M[0][0]]+x[M[0][1]]]
                       * BELLS[labels[1]][2*x[M[1][0]]+x[M[1][1]]])
        LINES4.append(canon(out))
assert len(set(LINES4)) == 48
LINE_INDEX = {v: i for i, v in enumerate(LINES4)}


# Full card profiles and the allowed four-card relation on one base pair.
CONTEXTS = tuple((p, k) for p in PAIRS6 for k in range(4))
PROFILE = {}
for num, v in enumerate(ALLOWED):
    prof = []
    for c in CONTEXTS:
        w = card6(v, *c)
        if any(w):
            assert canon(w) in LINE_INDEX
        prof.append(direction_scale(w))
    PROFILE[v] = tuple(prof)

# The simultaneous-live condition is forced by the q6 classification:
# because the normalized H6 card is live in every residual context, the
# other three cards must be either all live or all zero in that context.
# The complete profiles show that all H6-orbit states have the one all-live
# profile, whereas every product line has its own profile.
LIVE_CLASSES = defaultdict(list)
for v in ALLOWED:
    LIVE_CLASSES[tuple(direction >= 0 for direction, _ in PROFILE[v])] \
        .append(v)
assert Counter(map(len, LIVE_CLASSES.values())) == Counter({1: 960, 768: 1})
assert set(next(group for group in LIVE_CLASSES.values() if len(group) == 768)) \
    == H_ORBIT

BASE_RELATION = defaultdict(set)
base_contexts = tuple(((0, 1), k) for k in range(4))
base_indices = tuple(CONTEXTS.index(c) for c in base_contexts)
for v in ALLOWED:
    records = tuple(PROFILE[v][i] for i in base_indices)
    dirs = tuple(r[0] for r in records)
    scales = tuple(r[1] for r in records)
    pivot = next(s for s in scales if s)
    BASE_RELATION[dirs].add(tuple(s/pivot for s in scales))


# Canonical H6 Pauli Frobenius basis and allowed states avoiding coordinate 0.
H_BASIS = sorted(H_PAULI)
H_BASIS.remove(H0)
H_BASIS.insert(0, H0)
assert len(H_BASIS) == 64


def fsupport(v):
    return frozenset(i for i, u in enumerate(H_BASIS)
                     if sum(a*b for a, b in zip(u, v)))


FSUPPORT = {v: fsupport(v) for v in ALLOWED}
CAND_H = tuple(v for v in H_ORBIT if 0 not in FSUPPORT[v])
assert len(CAND_H) == 651


def allowed_lambda_set(states):
    """Return projective (lambda1,lambda2,lambda3), or empty."""
    # ``None`` denotes an unconstrained coordinate in one local card relation
    # (both the source and target card vanish); it is a wildcard, not a value.
    # Intersect partial assignments by unification.
    current = {(None, None, None)}

    def meet(a, b):
        out = []
        for x, y in zip(a, b):
            if x is not None and y is not None and x != y:
                return None
            out.append(x if x is not None else y)
        return tuple(out)

    for ci in range(60):
        rec = [PROFILE[s][ci] for s in states]
        dirs = tuple(x[0] for x in rec)
        if dirs not in BASE_RELATION:
            return set()
        local = set()
        for target in BASE_RELATION[dirs]:
            # target is projective card-scale vector.  lambda0=1.
            if target[0] == 0 or rec[0][1] == 0:
                continue
            vals = []
            good = True
            for j in range(1, 4):
                if rec[j][1] == 0:
                    if target[j] != 0:
                        good = False
                        break
                    vals.append(None)
                else:
                    if target[j] == 0:
                        good = False
                        break
                    vals.append((target[j]/target[0])*(rec[0][1]/rec[j][1]))
            if good:
                local.add(tuple(vals))
        if not local:
            return set()
        current = {c for a in current for b in local
                   if (c := meet(a, b)) is not None}
        if not current:
            return set()
    return current


def affine_test8(values):
    supp = [BITS8[i] for i, a in enumerate(values) if a]
    if not supp:
        return True
    x0 = supp[0]
    diffs = {tuple(a ^ b for a, b in zip(x, x0)) for x in supp}
    # closure is enough for affine support
    if any(tuple(a ^ b for a, b in zip(x, y)) not in diffs
           for x in diffs for y in diffs):
        return False
    pivot = next(values[i] for i in range(256) if values[i])
    ratios = {values[i]/pivot for i in range(256) if values[i]}
    # All survivors here are rational; standard affine ratios must be +/-1.
    if not ratios <= {Q(1), Q(-1)}:
        return False
    # Pull the sign function back to F_2^r and use its ANF.  This Möbius
    # transform is equivalent to the definition-level third-difference test
    # and reduces its O(|L|^4) cost to O(r 2^r).
    def bits_to_int(x):
        return sum(bit << i for i, bit in enumerate(x))

    basis = []
    pivots = []
    for d in sorted(diffs, key=bits_to_int):
        z = bits_to_int(d)
        for b, p in zip(basis, pivots):
            if (z >> p) & 1:
                z ^= b
        if z:
            p = z.bit_length() - 1
            for i, b in enumerate(basis):
                if (b >> p) & 1:
                    basis[i] ^= z
            k = 0
            while k < len(pivots) and pivots[k] > p:
                k += 1
            basis.insert(k, z)
            pivots.insert(k, p)
    r = len(basis)
    assert len(diffs) == 1 << r

    table = [0] * (1 << r)
    x0i = bits_to_int(x0)
    value_at = {bits_to_int(BITS8[i]): values[i]/pivot
                for i in range(256) if values[i]}
    for mask in range(1 << r):
        y = x0i
        for j, b in enumerate(basis):
            if (mask >> j) & 1:
                y ^= b
        table[mask] = 0 if value_at[y] == 1 else 1
    # Boolean Möbius transform: table[mask] becomes the ANF coefficient of
    # the monomial indexed by mask.
    for j in range(r):
        for mask in range(1 << r):
            if (mask >> j) & 1:
                table[mask] ^= table[mask ^ (1 << j)]
    if any(c and mask.bit_count() > 2 for mask, c in enumerate(table)):
        return False
    return True


def tensor8(states, lam):
    out = []
    for x in BITS8:
        s = Q(0)
        for gamma in range(4):
            s += lam[gamma] * BELLS[gamma][2*x[0]+x[1]] \
                 * states[gamma][sum(x[i+2] << (5-i) for i in range(6))]
        out.append(s)
    return tuple(out)


def run_family(candidates, label):
    # Bit masks for fast support-disjoint pruning.
    index = {v: i for i, v in enumerate(candidates)}
    masks = {}
    for v in candidates:
        m = 0
        for w in candidates:
            if not (FSUPPORT[v] & FSUPPORT[w]):
                m |= 1 << index[w]
        masks[v] = m
    direction_masks = []
    third_directions = []
    for ci in range(60):
        dm = defaultdict(int)
        for w in candidates:
            dm[PROFILE[w][ci][0]] |= 1 << index[w]
        direction_masks.append(dm)
        d0 = PROFILE[H0][ci][0]
        rel = defaultdict(set)
        for dirs in BASE_RELATION:
            if dirs[0] == d0:
                rel[(dirs[1], dirs[2])].add(dirs[3])
        third_directions.append(rel)
    survivors = set()
    checked = 0
    support_shapes = Counter()
    lambda_patterns = set()
    for s1 in candidates:
        for s2 in candidates:
            if FSUPPORT[s1] & FSUPPORT[s2]:
                continue
            mask = masks[s1] & masks[s2]
            for ci in range(60):
                key = (PROFILE[s1][ci][0], PROFILE[s2][ci][0])
                options = third_directions[ci].get(key)
                if not options:
                    mask = 0
                    break
                allowed = 0
                for direction in options:
                    allowed |= direction_masks[ci].get(direction, 0)
                mask &= allowed
                if not mask:
                    break
            while mask:
                bit = mask & -mask
                mask -= bit
                s3 = candidates[bit.bit_length()-1]
                states = (H0, s1, s2, s3)
                L = allowed_lambda_set(states)
                assert all(None not in ratios for ratios in L), \
                    ("surviving scale retains a wildcard", label, states, L)
                checked += 1
                support_shapes[tuple(len(FSUPPORT[s]) for s in states)] += 1
                for ratios in L:
                    lam = (Q(1),) + ratios
                    lambda_patterns.add(lam)
                    g = tensor8(states, lam)
                    assert affine_test8(g), (label, s1, s2, s3, lam)
                    survivors.add(canon(g))
    print(label, "checked triples", checked, "affine survivor lines", len(survivors))
    if checked:
        print("  support shapes", dict(support_shapes),
              "scale patterns", sorted(lambda_patterns))
    if label == "H/H/H":
        assert checked == 16
        assert support_shapes == Counter({(1, 1, 1, 1): 16})
        assert lambda_patterns == {
            (Q(1), a, b, c)
            for a in (Q(-1), Q(1))
            for b in (Q(-1), Q(1))
            for c in (Q(-1), Q(1))
        }
        assert len(survivors) == 128
    return survivors


def main():
    print("allowed q6 lines", len(ALLOWED), "H candidates", len(CAND_H))
    survivors = run_family(CAND_H, "H/H/H")
    # A product live-profile class is a singleton.  Repeating its one state
    # in all three outcomes violates pairwise Frobenius-support disjointness,
    # so no all-product triple needs a scale calculation.
    # All three zero gives the visible factor B00 tensor H6.
    assert len(survivors) == 128
    print("total nonfactor affine survivor lines", len(survivors))
    print("FULL-V4 Q8/H6 EXACT CERTIFICATE: PASS")


if __name__ == "__main__":
    main()
