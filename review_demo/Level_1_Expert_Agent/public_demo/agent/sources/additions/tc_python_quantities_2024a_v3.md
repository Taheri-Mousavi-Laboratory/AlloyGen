# TC-Python 2024a: equilibrium quantities and composition units

Author-prepared API definitions below, checked against the public
[2024a guide](https://thermocalc.com/wp-content/uploads/Documentation/Archive/2020-2024/2024a/2024a-tc-python-api-programmer-guide.pdf),
sections 5.5 (ThermodynamicQuantity) and 5.6 (CompositionUnit), and installed 2024a signatures.

Use `ThermodynamicQuantity` methods for equilibrium conditions/results:

| Method | Meaning |
| --- | --- |
| `temperature()` | Temperature, K |
| `pressure()` | Pressure, Pa |
| `system_size()` | System amount, mol |
| `mass_fraction_of_a_component(component)` | Component mass / system mass |
| `mole_fraction_of_a_component(component)` | Component moles / system moles |
| `mass_fraction_of_a_phase(phase)` | Phase mass / system mass |
| `mole_fraction_of_a_phase(phase)` | Phase molar fraction |
| `volume_fraction_of_a_phase(phase)` | Phase volume / system volume |
| `composition_of_phase_as_weight_fraction(phase, component)` | Component mass fraction within a phase |
| `composition_of_phase_as_mole_fraction(phase, component)` | Component mole fraction within a phase |
| `activity_of_component(component)` | Component activity |
| `chemical_potential_of_component(component)` | Component chemical potential |
| `gibbs_energy_of_a_phase(phase)` | Phase Gibbs energy |

Component composition, composition within a phase, and phase fraction describe different ratios.
Choose each input/output basis explicitly; matching numerical values do not make different bases equivalent.
Fractions are dimensionless; percent values are 100 times fractions on the same basis.

For interfaces accepting `CompositionUnit`, the available choices are
`MASS_PERCENT`, `MASS_FRACTION`, `MOLE_PERCENT`, and `MOLE_FRACTION`.
Use the documented unit accepted by each interface.

## Weight percent, mole percent and basis conversion

For component i, m_i is mass, n_i is amount in mol, and M_i is molar mass:

- Mass fraction: w_i = m_i / sum_j(m_j). A value p in wt% means w_i = p / 100.
- Mole fraction: x_i = n_i / sum_j(n_j). A value q in mol% means x_i = q / 100.

These follow the IUPAC definitions of [mass fraction](https://goldbook.iupac.org/terms/view/M03722),
[mole fraction](https://goldbook.iupac.org/terms/view/A00296) and
[percent](https://goldbook.iupac.org/terms/view/P04483).

Changing basis requires molar masses. From n_i = m_i / M_i:

- Mass to mole: x_i = (w_i / M_i) / sum_j(w_j / M_j).
- Mole to mass: w_i = (x_i * M_i) / sum_j(x_j * M_j).

The sums include all components, including the balance component. Use consistent
molar-mass units and a complete composition on one basis. Do not sum mass and
mole fractions together to determine the balance.

For native single-equilibrium conditions, wt% / 100 maps to
`mass_fraction_of_a_component`; mol% / 100 maps to `mole_fraction_of_a_component`.
Percent-to-fraction scaling preserves the basis. Check calculated compositions
in the bases originally requested.

## Composition conditions

Single-equilibrium conditions can combine mole-fraction constraints for some
components with mass-fraction constraints for others. Use each requested basis
directly and leave a balance component unconstrained; manual conversion to a
common basis is unnecessary.

## Numerical workaround — verify applicability

If an equilibrium calculation fails to converge at an exactly zero component
mass fraction, consider a small positive approximation. For example, try 1e-8
and check sensitivity against 1e-10. These values are mass fractions, not
percentages. Record the nominal and effective compositions. Accept the
approximation only if both calculations converge and the requested outputs
agree within a stated tolerance; otherwise report the unresolved failure.
