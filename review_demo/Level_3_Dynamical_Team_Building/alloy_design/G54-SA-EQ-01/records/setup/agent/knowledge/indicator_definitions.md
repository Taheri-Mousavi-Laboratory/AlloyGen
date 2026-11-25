# Alloy design calculation tools

Supply candidate compositions as a list of dictionaries containing `Ni`, `Er`
and `Zr` in **at.%**. Al is the balance and Y/Yb are zero. Each tool includes the
same fixed comparison alloy in its `reference` result, supplied internally in
**wt.%**. Compare the calculated indicators, keeping each composition's basis.
An empty list requests the reference only. Each calculation batch allows at most 3 candidates; the task states the whole-run allowances.
Repeated requests reuse matching saved calculations.

## Coarsening calculation

Use `evaluate_coarsening(compositions)` to compare L1₂ coarsening at 250°C.
It returns `J = m / m_reference`, where `m = absolute lattice misfit / diffusion
resistivity` and resistivity sums squared phase-composition differences divided
by the fixed diffusivities. **Lower J means greater modeled coarsening
resistance**; the reference is 1. Values are uncapped. This indicator uses the
fixed 250°C parameter model. An equilibrium without the required FCC/L1₂ pair
has an unavailable metric, not a favorable zero.

## Phase stability calculation

Use `evaluate_phase_stability(compositions)` for equilibrium L1₂ and competing
D0₂₃-Al₃Zr amounts at 250/660°C, in mol% of the whole system. More L1₂ at service
temperature supports strengthening potential; limiting high-temperature L1₂
serves a different phase-selection objective. Apply the task's priorities and
limits when comparing these quantities. Phase amounts alone do not predict size
or strength.

`Al3Zr_boundary` reports the **upper equilibrium disappearance temperature**
of D0₂₃-Al₃Zr. Read `liquid_at_upper_boundary` and any lower disappearance
intervals. A boundary involving liquid is not a solid-state heat-treatment
solvus. Lower competing-phase amount/stability can constrain Zr addition, but
must be weighed against its coarsening benefit. A boundary not detected within
the scanned range is reported as unavailable, never zero.

For this competing-phase objective, a higher Al₃Zr upper disappearance temperature is not a favorable direction. Include the liquid flag with that temperature. A phase absent at 250/660°C may still occur at other temperatures in the scan.

## Solidification calculation

Use `evaluate_solidification(compositions)` for classic Scheil solidification.
The tool returns terminal accumulated ternary Al₂₃Ni₆M₄, L1₂, Al₃Ni and Al₃Zr
amounts in mol% of the original system, plus the actual terminal solid amount.
These are solidification products, distinct from equilibrium phase amounts.
The ternary phase supplies a potential reservoir for subsequent precipitation;
it is not itself the final aged L1₂ fraction.

- `FR_K`: temperature at the saved total-solid mole fraction nearest 0.02,
  minus the actual terminal temperature.
- `CSC`: ratio of the 0.90–0.98 solid-fraction interval to the 0.40–0.90 interval
  on a heat-derived time coordinate. Heat is interpolated on a 0.001 solid-mole
  grid; the coordinate is `((H_start-H)/(H_start-H_end))²`.
- `HCS`: `CSC × FR_K / 100`, using the fixed source scaling.

Lower cracking indicators are preferred. They are screening proxies, not
cracking probabilities or experimental printability guarantees. A shorter
temperature interval does not by itself establish a shorter time interval.
Use the returned reference values with the same definitions; take any numerical
acceptance threshold from the task rather than assuming one.

## Using results

Inspect each result's status. `null` means unavailable; a physical zero requires
a completed calculation. Partial results retain usable indicators and explain
the missing ones. Full phase/heat curves and calculation details are linked in
`records`. Summaries do not choose candidates or resolve tradeoffs for you.
State which composition, indicator and reference support your decision.
