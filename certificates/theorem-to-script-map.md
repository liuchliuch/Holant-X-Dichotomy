# Manuscript-to-verifier map

The table links permanent LaTeX labels in the manuscript source to the finite
statement checked by each executable verifier. Each row identifies only the
finite substatement described in its middle column. An anchor introduced with
“used by” or “supports” locates a result that consumes that finite calculation;
it does not mean that the program proves the entire proposition, lemma, or
theorem.

| Manuscript anchor(s) | Checked finite statement | Verifier |
|---|---|---|
| `rem:sc-f6-core`, `rem:sc-f8-core` | Literal port-permutation/support dictionary identifying $H_6$ and $\mathrm{RM}_8$ with the Shao–Cai cores, and the normalized $(K^{-1})^{\otimes 8}$ identity | [`verify_sc20_core_dictionary.py`](verify_sc20_core_dictionary.py) |
| `prop:cert-equality-q4-boundary`, `eq:equality-q4-monomial-supports`, `eq:equality-q4-pauli-planes`, `eq:equality-q4-exotic-plane` | Stable equality-accessible q4 monomial supports, Pauli components, and exotic $V_4$ planes | [`verify_equality_q4.py`](verify_equality_q4.py) |
| `prop:cert-exotic-v4-q6-q8`; used by `lem:exotic-v4-q6`, `lem:exotic-v4-q8` | Complete normalized q6 survivor space and conditional all-four-live fixed-ruling q8 factor atlas | [`verify_exotic_v4_q6_q8.py`](verify_exotic_v4_q6_q8.py) |
| `prop:cert-twelve-loop`, `prop:cert-affine-q4-endpoint`; used by `prop:p1-affine-eight-vertex-certificate` | Deterministic twelve-loop partition, affine endpoint-nondegenerate q4 for the 18 displayed cores, and full-support fourth-root odd-table residues | [`verify_affine_q4.py`](verify_affine_q4.py) |
| `prop:cert-nd-q6-support-lowering` | Normalized-dihedral q6 four-flat support atlas and injective four-port support projection | [`verify_normalized_dihedral_q6.py`](verify_normalized_dihedral_q6.py) |
| `lem:app-v4-six-port` | Full $V_4$ q6 deck classification | [`verify_v4_q6.py`](verify_v4_q6.py) |
| `prop:cert-v4-q8-h6-parent`; used by `thm:v4-terminal-clean-q8` | Terminal-clean full $V_4$ q8 parent atlas with an $H_6$ card | [`verify_v4_q8_h6.py`](verify_v4_q8_h6.py) |
| `prop:cert-v4-five-spread`; used by `lem:v4-five-spread-gluing` | Full $V_4$ five-spread partial-live atlas | [`verify_v4_spread_partial.py`](verify_v4_spread_partial.py) |
| `prop:cert-v4-support-localization`; used by `lem:v4-spread-support-localization` | Five-spread clique and polar atlas | [`verify_v4_support_localization.py`](verify_v4_support_localization.py) |
| `prop:cert-v4-phase-localization`; used by `lem:v4-spread-phase-localization` | Affine-support (0/1/3/7) phase atlas | [`verify_v4_phase_localization.py`](verify_v4_phase_localization.py) |
| `prop:cert-platonic-incidence-atlas`, `prop:cert-platonic-q4-preservers` ($A_4/S_4$ portions), `prop:cert-platonic-q4-exits` ($A_4$ and first-form $S_4$ portions); used by `prop:p1-cert-A4-eight-vertex-orientation-terminal`, `prop:p1-cert-S4-uniform-eight-vertex-separator`, `lem:platonic-quaternary-boundary` | $A_4/S_4$ incidence, bridges, terminal partitions, and first-form minimum-q4 geometry | [`verify_a4_s4_certificates.py`](verify_a4_s4_certificates.py) |
| `prop:cert-platonic-incidence-atlas`, `prop:external-a4-q4-preservers` | External marked-$A_4$ normal form, deck incidence, and complete two-sided q4 rank profile | [`verify_a4_external_form.py`](verify_a4_external_form.py) |
| `prop:cert-A4-b8-parent-audit`; supports `prop:a4rel-required-eight-port` | $A_4$ hereditary q8 generated-parent audit | [`verify_a4_b8.py`](verify_a4_b8.py) |
| `prop:cert-platonic-q4-exits` (second-form $S_4$ portion); used by `prop:p1-cert-S4-uniform-eight-vertex-separator`, `lem:octahedral-separator` | $S_4$ second-form transported terminal partition and minimum-q4 terminals | [`verify_s4_second_form.py`](verify_s4_second_form.py) |
| `prop:cert-platonic-incidence-atlas`, `prop:cert-platonic-q4-exits` ($A_5$ portion); used by `lem:icosahedral-bridge`, `lem:icosahedral-separator` | Bounded $A_5$ incidence, bridge, and terminal audit | [`verify_a5_certificate.py`](verify_a5_certificate.py) |
| `prop:cert-platonic-incidence-atlas`, `prop:cert-platonic-deficient-mixed`, `prop:cert-platonic-q4-preservers` ($A_5/H_4$ portion); used by `lem:platonic-global-ruling` | Extended Platonic incidence, deletion connectivity, and $H_4$ completeness | [`verify_platonic_extended.py`](verify_platonic_extended.py) |

The proposition `prop:cert-platonic-incidence-atlas` has standard, external,
$A_5$, and extended components. Together, the four rows carrying that anchor
identify the verifier entry points for all of its finite components.
