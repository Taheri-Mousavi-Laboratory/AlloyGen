# Thermo-Calc quantity notation: composition, system and phase

Author-prepared summary, checked on 2026-10-27 against Thermo-Calc Software AB's
[public 2024a documentation set](https://thermocalc.com/wp-content/uploads/Documentation/Archive/2020-2024/2024a/2024a-thermo-calc-documentation-set.pdf#page=2222),
General Reference. The page column uses PDF viewer page numbers; the corresponding
General Reference pages are 8–21. Fractions below are dimensionless.

| Symbol | Definition | PDF page |
| --- | --- | --- |
| `X(component)` | Component mole fraction in the system | 2225, 2233 |
| `W(component)` | Component mass fraction in the system | 2225, 2233 |
| `X(phase,component)` | Component mole fraction within a phase | 2225, 2234 |
| `W(phase,component)` | Component mass fraction within a phase | 2225, 2235 |
| `T`, `P`, `N` | Temperature (K), pressure (Pa), total system amount (mol) | 2222, 2225 |
| `NP(phase)` | Phase amount (mol) | 2225 |
| `NPM(phase)` | Phase mole fraction: phase moles / system moles | 2234 |
| `BPW(phase)` | Phase mass fraction: phase mass / system mass | 2234 |
| `VPV(phase)` | Phase volume fraction: phase volume / system volume | 2234 |
| `G`, `H`, `U` | System Gibbs energy, enthalpy, internal energy (J) | 2223 |
| `GM`, `HM`, `UM` | Corresponding system molar quantities (J/mol) | 2223, 2230 |
| `S`, `SM` | System entropy (J/K), molar entropy (J/(mol K)) | 2223, 2230 |
| `V`, `VM` | System volume (m³), molar volume (m³/mol) | 2223, 2230 |

Scope matters: a quantity for the whole system differs from one for a named phase.
For these system quantities, the `M` suffix divides by total system moles
(PDF p. 2230). Thus numerical equality at a system amount of 1 mol does not make
extensive and molar units interchangeable. Percent equals 100 times the fraction
on the same composition basis (PDF pp. 2249–2250).
