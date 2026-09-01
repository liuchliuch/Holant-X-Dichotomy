#!/usr/bin/env python3
"""Replay the hereditary A4 eight-port certificate exactly.

The verifier reconstructs the H6 and matching six-port orbits, enumerates
the rich-line bridges and the resulting eight-port parents, and audits every
residual card.  Arithmetic and digest checks are exact.  The six-port orbits
are rebuilt in memory on every run; no persistent cache is read or written.
The ``--fresh`` flag is accepted to match the paper's replay command.
"""

from __future__ import annotations

import argparse
import itertools
from collections import deque
from fractions import Fraction

if not __debug__:
    raise SystemExit(
        "Assertions must be enabled; do not run this verifier with python -O."
    )

G = tuple[int, int]
Z: G = (0, 0)
O: G = (1, 0)
J: G = (0, 1)


def ga(a: G, b: G) -> G: return a[0] + b[0], a[1] + b[1]
def gn(a: G) -> G: return -a[0], -a[1]
def gs(k: int, a: G) -> G: return k * a[0], k * a[1]
def gm(a: G, b: G) -> G:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def nearest(q: Fraction) -> int:
    f = q.numerator // q.denominator
    return f + (q - f > Fraction(1, 2))


def divmod_g(a: G, b: G) -> tuple[G, G]:
    n = b[0] * b[0] + b[1] * b[1]
    qr = nearest(Fraction(a[0] * b[0] + a[1] * b[1], n))
    qi = nearest(Fraction(a[1] * b[0] - a[0] * b[1], n))
    q = qr, qi
    p = gm(q, b)
    return q, (a[0] - p[0], a[1] - p[1])


def gcd_g(a: G, b: G) -> G:
    while b != Z:
        _, r = divmod_g(a, b)
        a, b = b, r
    return a


def exact_div_g(a: G, b: G) -> G:
    q, r = divmod_g(a, b)
    assert r == Z
    return q


def canonical(v: tuple[G, ...]) -> tuple[G, ...]:
    d = Z
    for x in v:
        if x != Z:
            d = x if d == Z else gcd_g(d, x)
    assert d != Z
    p = tuple(exact_div_g(x, d) for x in v)
    units = (O, (-1, 0), J, (0, -1))
    return min(tuple(gm(u, x) for x in p) for u in units)


Q0 = (O, Z, Z, O)
Q1 = (Z, J, J, Z)
Q2 = (Z, O, (-1, 0), Z)
Q3 = (J, Z, Z, (0, -1))
R = tuple(ga(ga(Q0[k], Q1[k]), ga(Q2[k], Q3[k])) for k in range(4))


def bell(label: int) -> tuple[G, ...]:
    d, ell = divmod(label, 2)
    return tuple(((-1 if ell * x & 1 else 1), 0) if (x ^ y) == d else Z
                 for x in (0, 1) for y in (0, 1))


B = tuple(bell(k) for k in range(4))
THETA = ((0, 1), (1, 1))


def theta(x: int) -> int:
    d, l = divmod(x, 2)
    return 2 * l + (d ^ l)


def h6() -> tuple[G, ...]:
    out = []
    for bits in itertools.product((0, 1), repeat=6):
        value = Z
        for gamma in range(4):
            x = B[gamma][2 * bits[0] + bits[1]]
            y = B[theta(gamma)][2 * bits[2] + bits[3]]
            z = B[theta(theta(gamma))][2 * bits[4] + bits[5]]
            value = ga(value, gm(gm(x, y), z))
        out.append(value)
    return canonical(tuple(out))


def local(t: tuple[G, ...], port: int, M: tuple[G, ...]) -> tuple[G, ...]:
    out = [Z] * 64
    shift = 5 - port
    for word in range(64):
        y = (word >> shift) & 1
        base = word & ~(1 << shift)
        value = Z
        for x in (0, 1):
            value = ga(value, gm(M[2 * y + x], t[base | (x << shift)]))
        out[word] = value
    return canonical(tuple(out))


def swap(t: tuple[G, ...], port: int) -> tuple[G, ...]:
    out = [Z] * 64
    a, b = 5 - port, 5 - (port + 1)
    for word in range(64):
        wa, wb = (word >> a) & 1, (word >> b) & 1
        source = word if wa == wb else word ^ (1 << a) ^ (1 << b)
        out[word] = t[source]
    return canonical(tuple(out))


def orbit():
    start = h6()
    seen = {start}
    todo = deque([start])
    gens = (Q1, R)
    count = 0
    while todo:
        t = todo.popleft()
        count += 1
        for p in range(6):
            for M in gens:
                u = local(t, p, M)
                if u not in seen:
                    seen.add(u); todo.append(u)
        for p in range(5):
            u = swap(t, p)
            if u not in seen:
                seen.add(u); todo.append(u)
        if count % 10000 == 0:
            print('processed', count, 'seen', len(seen), flush=True)
    print('H6 orbit', len(seen))
    return seen


def tetra_points():
    out = [Q0, Q1, Q2, Q3]
    for e in itertools.product((-1, 1), repeat=3):
        out.append(tuple(ga(ga(Q0[k], gs(e[0], Q1[k])),
                            ga(gs(e[1], Q2[k]), gs(e[2], Q3[k])))
                         for k in range(4)))
    return tuple(out)


def matchings(ports):
    if not ports:
        yield (); return
    a = ports[0]
    for j in range(1, len(ports)):
        b = ports[j]
        for rest in matchings(ports[1:j] + ports[j+1:]):
            yield ((a, b), *rest)


def product6(matching, factors):
    out = []
    for bits in itertools.product((0, 1), repeat=6):
        value = O
        for (a, b), M in zip(matching, factors):
            value = gm(value, M[2 * bits[a] + bits[b]])
        out.append(value)
    return canonical(tuple(out))


def matching_orbit():
    T = tetra_points()
    out = set()
    for matching in matchings(tuple(range(6))):
        for factors in itertools.product(T, repeat=3):
            out.add(product6(matching, factors))
    print('matching orbit', len(out))
    return out


def line_key(fixed, point):
    pivot = next(j for j, x in enumerate(fixed) if x != Z)
    residual = tuple(ga(gm(point[j], fixed[pivot]),
                        gn(gm(fixed[j], point[pivot])))
                     for j in range(64))
    return canonical(residual)


def rich_lines(fixed, allowed):
    buckets = {}
    for count, point in enumerate(allowed, 1):
        if point == fixed:
            continue
        key = line_key(fixed, point)
        buckets.setdefault(key, []).append(point)
        if count % 10000 == 0:
            print('  line points', count, flush=True)
    hist = {}
    for values in buckets.values():
        hist[len(values)] = hist.get(len(values), 0) + 1
    rich = sorted(tuple(sorted(values)) for values in buckets.values()
                  if len(values) >= 2)
    print('line histogram', sorted(hist.items()), 'rich', len(rich))
    return rich


def permute(t, permutation):
    out=[]
    for y in itertools.product((0,1),repeat=6):
        x=[0]*6
        for old,new in enumerate(permutation):
            x[old]=y[new]
        word=sum(x[p]<<(5-p) for p in range(6))
        out.append(t[word])
    return canonical(tuple(out))


def local_pair_fix_matrix(L):
    T=tetra_points()
    target=canonical(Q0)
    for M in T:
        # L Q0 M^T
        out=[]
        for a,b in itertools.product((0,1),repeat=2):
            value=Z
            for x,y in itertools.product((0,1),repeat=2):
                value=ga(value,gm(gm(L[2*a+x],Q0[2*x+y]),M[2*b+y]))
            out.append(value)
        if canonical(tuple(out))==target:return M
    raise AssertionError


def line_orbits(base, lines, generators):
    lookup={frozenset(line):j for j,line in enumerate(lines)}
    perms=[]
    for transform in generators:
        assert transform(base)==base
        image=[]
        for line in lines:
            pair=frozenset(transform(x) for x in line)
            assert pair in lookup
            image.append(lookup[pair])
        perms.append(image)
    unseen=set(range(len(lines)));orbits=[]
    while unseen:
        seed=next(iter(unseen));orb={seed};todo=[seed]
        while todo:
            x=todo.pop()
            for p in perms:
                y=p[x]
                if y not in orb:orb.add(y);todo.append(y)
        unseen-=orb;orbits.append(orb)
    return sorted(map(len,orbits),reverse=True),orbits


def matching_stabilizer_generators():
    gens=[]
    for a,b in ((0,1),(2,3),(4,5)):
        gens.append(lambda t,a=a: swap(t,a))
        for L in (Q1,R):
            M=local_pair_fix_matrix(L)
            gens.append(lambda t,a=a,b=b,L=L,M=M: local(local(t,a,L),b,M))
    for perm in ((2,3,0,1,4,5),(0,1,4,5,2,3)):
        gens.append(lambda t,perm=perm:permute(t,perm))
    return gens


def pauli_stabilizer_generators(base):
    paulis=(Q0,Q1,Q2,Q3);fix=[]
    for matrices in itertools.product(paulis,repeat=6):
        t=base
        for p,M in enumerate(matrices):t=local(t,p,M)
        if t==base:fix.append(matrices)
    print('local Pauli stabilizers',len(fix))
    return [lambda t,m=m: __import__('functools').reduce(
                lambda state,pm:local(state,pm[0],pm[1]),enumerate(m),t)
            for m in fix]


def pair_action_option(L,M):
    labels=[];scalars=[]
    bellkeys=[canonical(x) for x in B]
    for bvec in B:
        out=[]
        for a,b in itertools.product((0,1),repeat=2):
            value=Z
            for x,y in itertools.product((0,1),repeat=2):
                value=ga(value,gm(gm(L[2*a+x],bvec[2*x+y]),M[2*b+y]))
            out.append(value)
        k=canonical(tuple(out))
        if k not in bellkeys:return None
        label=bellkeys.index(k);labels.append(label)
        pivot=next(j for j,z in enumerate(B[label]) if z!=Z)
        # Since the target Bell pivot is +/-1, this is the exact scalar.
        scalars.append(out[pivot] if B[label][pivot]==O else gn(out[pivot]))
    return tuple(labels),tuple(scalars),L,M


def h6_pair_stabilizers():
    options=[]
    for L,M in itertools.product(tetra_points(),repeat=2):
        o=pair_action_option(L,M)
        if o is not None:options.append(o)
    print('Bell-permuting A4 pair actions',len(options))
    fixes=[]
    target={(g,theta(g),theta(theta(g))) for g in range(4)}
    for oa,ob,oc in itertools.product(options,repeat=3):
        terms=[];weights=[]
        for g in range(4):
            terms.append((oa[0][g],ob[0][theta(g)],oc[0][theta(theta(g))]))
            weights.append(gm(gm(oa[1][g],ob[1][theta(g)]),oc[1][theta(theta(g))]))
        if set(terms)==target and all(w==weights[0] for w in weights):
            fixes.append((oa[2:],ob[2:],oc[2:]))
    print('pair-block H6 stabilizers',len(fixes))
    transforms=[]
    for blocks in fixes:
        def transform(t,blocks=blocks):
            for block,(L,M) in enumerate(blocks):
                t=local(local(t,2*block,L),2*block+1,M)
            return t
        transforms.append(transform)
    return transforms


def direct_permutation_stabilizers(base):
    fixes=[]
    for p in itertools.permutations(range(6)):
        if permute(base,p)==base:
            fixes.append(lambda t,p=p:permute(t,p))
    print('direct H6 port stabilizers',len(fixes))
    return fixes

class _ExploreNamespace:
    pass


e = _ExploreNamespace()
for _name in ['Z','Q0','Q1','Q2','Q3','R','canonical','ga','gm','gn','gs','h6','tetra_points','orbit','matching_orbit','matchings','product6','rich_lines','line_orbits','matching_stabilizer_generators','h6_pair_stabilizers','direct_permutation_stabilizers']:
    setattr(e, _name, globals()[_name])
del _name

# Eight-port bridge and exhaustive audit.
import hashlib
from collections import Counter
from math import lcm


QG = tuple[Fraction, Fraction]
V = tuple[QG, ...]
QZ: QG = (Fraction(0), Fraction(0))
QO: QG = (Fraction(1), Fraction(0))

EXPECTED_DIGESTS={
    'domain':'d9634d6cec551203408187e3880f9d1e719eab1af1a27c311a96ab2aedd86daa',
    'h6-orbit':'8a7d37ceff29cc2e070e2535f8a26849d544171857342aad5dc37db33098e1fc',
    'matching-orbit':'1b88bbf1009659afae87e6481e21429ba87edfb04b087f39176daa98856be9d8',
    'allowed':'4b6d6288e41ed33727ef4a4b70180f6ffc52c7076b8ba543cbe814b074cb9b99',
    'lines':'60762daae46528be622f1a78240656398ff74c3d53922496c86856763cb75105',
    'bridges':'e4c3f94585924a7a6251ef18d53def57826a7b3e1ef986af18b283da648127e0',
    'parents':'ed1357b63807b43ba26cf47a34cbc94af90637bdbeda376db6a2dac971dda044',
    'audit':'d4ce9882709f792002d949cd7db15888681861d054a8b7688332d7720f55b075',
}


def digest(records):
    h = hashlib.sha256()
    for record in sorted(records):
        h.update(repr(record).encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def checked_digest(name,records):
    value=digest(records)
    assert value==EXPECTED_DIGESTS[name],(name,value)
    print(name,'digest',value)
    return value


def qrecord(v):
    return tuple((a.numerator, a.denominator, b.numerator, b.denominator)
                 for a, b in v)


def vector_rank(vectors):
    rows=[list(v) for v in vectors if not vz(v)]
    rank=0
    for col in range(64):
        pivot=next((j for j in range(rank,len(rows)) if rows[j][col]!=QZ),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank]
        z=rows[rank][col]
        rows[rank]=[qd(x,z) for x in rows[rank]]
        for j in range(len(rows)):
            if j==rank or rows[j][col]==QZ:continue
            c=rows[j][col]
            rows[j]=[qa(x,qn(qm(c,y))) for x,y in zip(rows[j],rows[rank])]
        rank+=1
        if rank==len(rows):break
    return rank


def qa(x, y): return x[0] + y[0], x[1] + y[1]
def qn(x): return -x[0], -x[1]
def qm(x, y): return x[0]*y[0]-x[1]*y[1], x[0]*y[1]+x[1]*y[0]
def qi(x):
    d=x[0]*x[0]+x[1]*x[1]
    return x[0]/d,-x[1]/d
def qd(x,y): return qm(x,qi(y))
def vs(c,v): return tuple(qm(c,x) for x in v)
def va(x,y): return tuple(qa(a,b) for a,b in zip(x,y))
def vn(x): return tuple(qn(a) for a in x)
def vz(x): return all(a==QZ for a in x)
def qv(x): return tuple((Fraction(a),Fraction(b)) for a,b in x)


def key(v: V):
    den=1
    for a,b in v: den=lcm(den,a.denominator,b.denominator)
    z=tuple((int(a*den),int(b*den)) for a,b in v)
    return e.canonical(z)


def allowed_vector(v: V, allowed: set) -> bool:
    return vz(v) or key(v) in allowed


def solve(left:V,right:V,target:V):
    for i in range(64):
      for j in range(i+1,64):
        det=qa(qm(left[i],right[j]),qn(qm(left[j],right[i])))
        if det!=QZ:
          a=qd(qa(qm(target[i],right[j]),qn(qm(target[j],right[i]))),det)
          b=qd(qa(qm(left[i],target[j]),qn(qm(left[j],target[i]))),det)
          assert va(vs(a,left),vs(b,right))==target
          return a,b
    raise ValueError


def scaled_opposite(base,p,q):
    a,b=solve(qv(p),qv(q),vs((Fraction(2),Fraction(0)),qv(base)))
    return vs(a,qv(p)),vs(b,qv(q))


def bridge(A:V,D:V,allowed:set, linesA):
    out={tuple(QZ for _ in range(64))}
    # independent X and A-2X points on a rich line through A
    for line in linesA:
      for px,py in (line,line[::-1]):
        # A = 2X + Y
        a,b=solve(qv(px),qv(py),A)
        X=vs((a[0]/2,a[1]/2),qv(px))
        if allowed_vector(va(D,vs((Fraction(2),Fraction(0)),X)), allowed):
          out.add(X)
    # X parallel A: choose Z=D+2X among the two other points of line AD.
    Adir=key(A); Ddir=key(D)
    adlines=[line for line in linesA if Ddir in line]
    assert len(adlines)==1
    for zdir in adlines[0]:
        if zdir==Ddir:continue
        # D + 2x A = z zdir
        x,z=solve(qv(Adir),qv(zdir),vn(D))
        # x*A+z*zdir=-D, hence D+x*A=-z*zdir
        X=vs((x[0]/2,x[1]/2),qv(Adir))
        if allowed_vector(va(A,vn(vs((Fraction(2),Fraction(0)),X))), allowed):
            out.add(X)
    return out


def triples(B,S):
    by={}
    for x in B:
      for y in B: by.setdefault(va(x,y),0);by[va(x,y)]+=1
    count=0
    for z in B:
      need=va(S,vn(z));count+=by.get(need,0)
    return count


def triple_list(B,S):
    B=list(B);by={}
    for x in B:
      for y in B:by.setdefault(va(x,y),[]).append((x,y))
    out=[]
    for z in B:
      for x,y in by.get(va(S,vn(z)),[]):out.append((x,y,z))
    return out


def tensor8(columns):
    Q0,Q1,Q2,Q3=map(qv,(e.Q0,e.Q1,e.Q2,e.Q3))
    dual=(vs((Fraction(1,2),Fraction(0)),Q0),
          vs((Fraction(-1,2),Fraction(0)),Q1),
          vs((Fraction(1,2),Fraction(0)),Q2),
          vs((Fraction(-1,2),Fraction(0)),Q3))
    out=[]
    for word in range(256):
      a=(word>>7)&1;b=(word>>6)&1;r=word&63
      z=QZ
      for mu in range(4):z=qa(z,qm(dual[mu][2*a+b],columns[mu][r]))
      out.append(z)
    return tuple(out)


def contract8(t,pair,K):
    remain=[p for p in range(8) if p not in pair];out=[]
    for y in itertools.product((0,1),repeat=6):
      z=QZ
      for a,b in itertools.product((0,1),repeat=2):
        bits=[0]*8;bits[pair[0]]=a;bits[pair[1]]=b
        for p,c in zip(remain,y):bits[p]=c
        word=sum(bits[p]<<(7-p) for p in range(8))
        z=qa(z,qm(qv(K)[2*a+b],t[word]))
      out.append(z)
    return tuple(out)


def globally_safe(t,allowed):
    for pair in itertools.combinations(range(8),2):
      for K in e.tetra_points():
        out=contract8(t,pair,K)
        if not vz(out) and key(out) not in allowed:return False,pair,K,out
    return True,None,None,None


def xor_basis(space):
    basis=[]
    for x in sorted(space):
      y=x
      for b in basis:y=min(y,y^b)
      if y:basis.append(y);basis.sort(reverse=True)
    return basis


def affine(t):
    supp=[j for j,z in enumerate(t) if z!=QZ]
    if not supp:return True
    o=supp[0];L={x^o for x in supp}
    if any(x^y not in L for x in L for y in L):return False
    basis=xor_basis(L);coord={}
    for mask in range(1<<len(basis)):
      x=0
      for j,b in enumerate(basis):
        if mask>>j&1:x^=b
      coord[x]=mask
    roots=(QO,(Fraction(0),Fraction(1)),(Fraction(-1),Fraction(0)),(Fraction(0),Fraction(-1)))
    base=t[o];ex={}
    for x in supp:
      vals=tuple(qm(base,r) for r in roots)
      if t[x] not in vals:return False
      ex[coord[x^o]]=vals.index(t[x])
    d=len(basis);lin=[ex[1<<j] for j in range(d)];quad={}
    for j in range(d):
      for k in range(j+1,d):
        z=(ex[(1<<j)|(1<<k)]-lin[j]-lin[k])%4
        if z not in (0,2):return False
        quad[j,k]=z
    for mask,z in ex.items():
      q=sum(lin[j] for j in range(d) if mask>>j&1)
      q+=sum(c for (j,k),c in quad.items() if mask>>j&1 and mask>>k&1)
      if q%4!=z:return False
    return True


def factor_cut(t):
    ports=range(8)
    for size in range(1,5):
      for S in itertools.combinations(ports,size):
        if size==4 and 0 not in S:continue
        S=set(S);T=[p for p in ports if p not in S]
        rows=[]
        for a in itertools.product((0,1),repeat=len(S)):
          row=[]
          for b in itertools.product((0,1),repeat=len(T)):
            bits=[0]*8
            for p,c in zip(sorted(S),a):bits[p]=c
            for p,c in zip(T,b):bits[p]=c
            row.append(t[sum(bits[p]<<(7-p) for p in ports)])
          rows.append(row)
        pivot=next(((i,j) for i,row in enumerate(rows) for j,z in enumerate(row) if z!=QZ),None)
        if pivot is None:return tuple(sorted(S))
        i0,j0=pivot;z0=rows[i0][j0]
        if all(qm(rows[i][j],z0)==qm(rows[i][j0],rows[i0][j])
               for i in range(len(rows)) for j in range(len(rows[0]))):
            return tuple(sorted(S))
    return None


def domain_line_certificate():
    T=e.tetra_points(); keys=[e.canonical(x) for x in T]
    def mm(A,B):
      out=[]
      for a,b in itertools.product((0,1),repeat=2):
        z=e.Z
        for x in (0,1):z=e.ga(z,e.gm(A[2*a+x],B[2*x+b]))
        out.append(z)
      return tuple(out)
    def adj(A):return A[3],e.gn(A[1]),e.gn(A[2]),A[0]
    def index(A):return keys.index(e.canonical(A))
    generated={e.canonical(e.Q0)};todo=list(generated)
    while todo:
      A=todo.pop()
      for U in (e.Q1,e.R):
        B=e.canonical(mm(A,U))
        if B not in generated:generated.add(B);todo.append(B)
    assert generated==set(keys) and len(generated)==12
    lines=[]
    for signs in ((1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)):
      ends=[]
      for s in (signs,tuple(-x for x in signs)):
        ends.append(tuple(e.ga(e.ga(e.Q0[k],e.gs(s[0],e.Q1[k])),
                                     e.ga(e.gs(s[1],e.Q2[k]),e.gs(s[2],e.Q3[k])))
                          for k in range(4)))
      assert tuple(e.ga(ends[0][k],ends[1][k]) for k in range(4)) == \
             tuple(e.gs(2,x) for x in e.Q0)
      lines.append(frozenset(map(index,ends)))
    assert len(set(lines))==4
    permutations=[]
    for U in T:
      p=[]
      for line in lines:
        image=frozenset(index(mm(mm(U,T[j]),adj(U))) for j in line)
        p.append(lines.index(image))
      permutations.append(tuple(p))
    assert len(set(permutations))==12
    assert {p[0] for p in permutations}==set(range(4))
    record=(tuple(sorted(generated)),tuple(tuple(sorted(x)) for x in lines),
            tuple(sorted(set(permutations))))
    print('generator closure',len(generated))
    print('domain lines',len(lines),'conjugation permutations',len(set(permutations)))
    checked_digest('domain',[record])
    return record


def representatives(base,lines,gens):
    _,orbits=e.line_orbits(base,lines,gens)
    data=[]
    for orbit0 in orbits:
      j=min(orbit0,key=lambda k:tuple(sorted(lines[k])))
      data.append((tuple(sorted(lines[j])),lines[j],len(orbit0)))
    data.sort()
    return [x[1] for x in data],[x[2] for x in data]


def main():
    parser = argparse.ArgumentParser(
        description="Replay the hereditary A4 eight-port certificate exactly."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="explicitly request fresh orbit reconstruction (the current default)",
    )
    parser.parse_args()

    domain_line_certificate()
    ho=e.orbit();mo=e.matching_orbit()
    assert len(ho)==31104 and len(mo)==25920 and ho.isdisjoint(mo)
    allowed=ho|mo
    checked_digest('h6-orbit',ho)
    checked_digest('matching-orbit',mo)
    checked_digest('allowed',allowed)
    mb=e.product6(next(e.matchings(tuple(range(6)))),(e.Q0,e.Q0,e.Q0))
    bases=[('M',mb,e.matching_stabilizer_generators()),
           ('H',e.h6(),[*e.h6_pair_stabilizers(),*e.direct_permutation_stabilizers(e.h6())])]
    candidates=[]; bridge_records=[]; line_records=[]
    for bt,base,gens in bases:
      print('BASE',bt,flush=True)
      lines=e.rich_lines(base,allowed)
      type_hist=Counter(''.join(sorted('H' if x in ho else 'M' for x in line))
                        for line in lines)
      print(' endpoint types',type_hist)
      if bt=='M':assert type_hist==Counter({'MM':48,'HH':36})
      else:assert type_hist==Counter({'HM':60,'HH':24})
      line_records.append((bt, tuple(sorted(tuple(sorted(line)) for line in lines))))
      reps,osizes=representatives(base,lines,gens)
      print(' line orbit sizes',osizes)
      if bt=='M':assert sorted(osizes)==[12,36,36]
      else:assert sorted(osizes)==[4,8,12,12,12,12,24]
      for oi,line in enumerate(reps):
        typ=''.join(sorted('H' if x in ho else 'M' for x in line))
        A,D=scaled_opposite(base,*line)
        print(' line orbit',oi,typ,'finding lines through A',flush=True)
        linesA=e.rich_lines(key(A),allowed)
        B=bridge(A,D,allowed,linesA)
        S=vs((Fraction(1,2),Fraction(0)),va(A,vn(D)))
        triples0=triple_list(B,S)
        print(' bridge',len(B),'triples',len(triples0),flush=True)
        assert len(B)==6 and len(triples0)==12
        bridge_records.append((bt,oi,typ,qrecord(A),qrecord(D),
                               tuple(sorted(qrecord(x) for x in B)),
                               tuple(sorted(tuple(qrecord(x) for x in triple)
                                            for triple in triples0))))
        for orient in (1,-1):
          for triple in triples0:
            cols=(qv(base),)+tuple(x if orient==1 else vn(x) for x in triple)
            candidates.append((bt,oi,typ,orient,tensor8(cols)))
    print('candidate maps',len(candidates))
    assert len(candidates)==240
    # Projective deduplication before the expensive 336-card audit.
    unique={key(t):(bt,oi,typ,o,t) for bt,oi,typ,o,t in candidates}
    print('unique parents',len(unique))
    assert len(unique)==240
    checked_digest('lines',line_records)
    checked_digest('bridges',bridge_records)
    checked_digest('parents',unique)
    hist=Counter();rank_hist=Counter();survivors=[]
    audit_records=[]
    for number,k in enumerate(sorted(unique),1):
      item=unique[k]
      bt,oi,typ,o,t=item
      columns=tuple(contract8(t,(0,1),K) for K in (e.Q0,e.Q1,e.Q2,e.Q3))
      r=vector_rank(columns)
      safe,*w=globally_safe(t,allowed)
      if not safe:
        hist['actual q6 exit']+=1
        rank_hist[(r,'actual q6 exit')]+=1
        pair,K,out=w
        audit_records.append((k,'exit',r,pair,e.canonical(K),key(out)))
      else:
        a=affine(t);cut=factor_cut(t)
        hist['affine' if a else ('factor' if cut is not None else 'UNRESOLVED')]+=1
        rank_hist[(r,'affine' if a else ('factor' if cut is not None else 'UNRESOLVED'))]+=1
        survivors.append((bt,oi,typ,o,k,a,cut))
        audit_records.append((k,'safe',r,a,cut))
      if number%25==0:print(' audited',number,hist,flush=True)
    print('hist',hist)
    print('rank hist',sorted(rank_hist.items()))
    print('survivors',len(survivors))
    assert hist==Counter({'actual q6 exit':180,'affine':60})
    assert rank_hist==Counter({(2,'actual q6 exit'):120,
                              (4,'actual q6 exit'):60,
                              (4,'affine'):60})
    assert len(survivors)==60 and sum(x[-1] is not None for x in survivors)==24
    checked_digest('audit',audit_records)
    print('A4 HEREDITARY Q8 EXACT CERTIFICATE: PASS')


if __name__=='__main__':main()
