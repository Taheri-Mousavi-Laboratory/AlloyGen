"""Reusable equilibrium and coarsening calculations for the supplied Al model.

No reference answers, composition selection, file writes or model calls.
TC-Python is imported only by calculate_equilibrium, after runtime configuration.
"""
import math

ELEMENTS = ("Al", "Ni", "Er", "Zr", "Y", "Yb")
SOLUTES = ELEMENTS[1:]

def coarsening(phases, parameters):
    matrix = [p for p in phases if p["name"].split("#")[0] == "FCC_A1" and p["mole_fraction"] > 1e-12]
    precip = [p for p in phases if p["name"].split("#")[0] == "AL3M_L12" and p["mole_fraction"] > 1e-12]
    if len(matrix) != 1 or len(precip) != 1:
        return {"status": "undefined", "reason": "Need one populated FCC matrix and one populated L12 composition set"}
    a, p = matrix[0]["composition_mole_fraction"], precip[0]["composition_mole_fraction"]
    solutes = ["Ni", "Er", "Zr", "Y", "Yb"]
    terms = {e: (p[e] - a[e]) ** 2 / parameters["Diffusivity"][e] for e in solutes}
    resistance = sum(terms.values())
    lattice = parameters["latticep"]
    a_matrix = sum(a[e] * lattice[e] for e in solutes) + (1 - sum(a[e] for e in solutes)) * lattice["Al"]
    a_precip = sum(p[e] * lattice[e] for e in solutes) + (1 - sum(p[e] for e in solutes)) * lattice["Al"]
    epsilon = 100 * abs(1 - a_precip / a_matrix)
    if resistance <= 0 or not all(math.isfinite(v) for v in [resistance, epsilon, a_matrix, a_precip]):
        return {"status": "undefined", "reason": "Nonpositive resistivity or nonfinite input"}
    return {"status": "computed_with_historical_parameters", "R_code": resistance,
            "resistivity_terms": terms, "a_matrix_angstrom": a_matrix,
            "a_L12_angstrom": a_precip, "misfit_percent": epsilon,
            "m_raw": epsilon / resistance, "m_historical_scaled": epsilon / (resistance * 1e-23),
            "L12_mole_fraction": precip[0]["mole_fraction"]}


def validate_composition(composition):
    """Require explicit solute mole fractions; Al is the positive balance."""
    if set(composition) != set(SOLUTES):
        raise ValueError('Specify Ni, Er, Zr, Y and Yb mole fractions explicitly')
    if any(not math.isfinite(v) or v < 0 or v >= 1 for v in composition.values()):
        raise ValueError('Solute mole fractions must be finite and in [0, 1)')
    if sum(composition.values()) >= 1:
        raise ValueError('Al balance must be positive')
    return dict(composition)


def phase_fraction(phases, name):
    """Sum populated composition sets of a phase from a complete result."""
    return sum(p['mole_fraction'] for p in phases if p['name'].split('#')[0] == name)


def normalized_coarsening(descriptor, reference_m):
    """Retain raw quantities; undefined descriptors stay null, never zero."""
    if not math.isfinite(reference_m) or reference_m <= 0:
        raise ValueError('Normalization reference must be finite and positive')
    if descriptor['status'] != 'computed_with_historical_parameters':
        return dict(descriptor, J=None, J_capped=None, improvement=None)
    m = descriptor['m_raw']
    if not math.isfinite(m) or m < 0:
        raise ValueError('Invalid coarsening descriptor')
    j = m / reference_m
    return dict(descriptor, J=j, J_capped=min(j, 4.), improvement=1/j if j > 0 else None)


def equilibrium_summary(record, parameters, reference_m):
    """Postprocess a complete saved or fresh equilibrium using one convention."""
    if record['status'] != 'complete':
        raise ValueError('Cannot summarize an incomplete equilibrium')
    if record['input_basis'] != 'mole_fraction':
        raise ValueError('This helper expects explicit bulk mole fractions')
    composition = validate_composition(record['input_solute_fractions'])
    phases = record['phases']
    expected = dict(composition, Al=1-sum(composition.values()))
    errors = {e: sum(p['mole_fraction']*p['composition_mole_fraction'][e] for p in phases)-value
              for e, value in expected.items()}
    values = [p['mole_fraction'] for p in phases]
    values += [x for p in phases for x in p['composition_mole_fraction'].values()]
    if not phases or not all(math.isfinite(v) for v in values):
        raise ValueError('Missing or nonfinite equilibrium quantities')
    if record['temperature_K'] == parameters['temperature_K']:
        if any(not math.isfinite(parameters['Diffusivity'][e]) or parameters['Diffusivity'][e] <= 0 for e in SOLUTES):
            raise ValueError('Diffusivities must be finite and positive')
        if any(not math.isfinite(parameters['latticep'][e]) or parameters['latticep'][e] <= 0 for e in ELEMENTS):
            raise ValueError('Lattice parameters must be finite and positive')
        descriptor = coarsening(phases, parameters)
    else:
        descriptor = {'status':'not_evaluated', 'reason':'Historical descriptor parameters apply only at their recorded temperature'}
    return dict(phase_mole_fractions={name:phase_fraction(phases,name) for name in sorted({p['name'].split('#')[0] for p in phases})},
                component_balance_errors=errors, phase_sum_error=sum(p['mole_fraction'] for p in phases)-1,
                max_phase_composition_sum_error=max(abs(sum(p['composition_mole_fraction'].values())-1) for p in phases),
                minimum_fraction=min(values), coarsening=normalized_coarsening(descriptor,reference_m))


def calculate_equilibrium(system, composition, temperature_K, pressure_Pa=101325):
    """One native-composition equilibrium; caller owns session and persistence.

    Errors propagate to the caller so it can preserve the failed attempt.
    The system must select ELEMENTS with the intended database/phases.
    """
    composition = validate_composition(composition)
    if not math.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError('Temperature must be positive and in kelvin')
    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0:
        raise ValueError('Pressure must be positive and in pascals')
    from tc_python import ThermodynamicQuantity as TQ
    calculation = system.with_single_equilibrium_calculation().set_condition('T',temperature_K).set_condition('P',pressure_Pa)
    for element, value in composition.items():
        calculation.set_condition(TQ.mole_fraction_of_a_component(element),value)
    result = calculation.calculate()
    phases = [{'name':name, 'mole_fraction':float(result.get_value_of(TQ.mole_fraction_of_a_phase(name))),
               'composition_mole_fraction':{e:float(result.get_value_of(TQ.composition_of_phase_as_mole_fraction(name,e))) for e in ELEMENTS}}
              for name in result.get_stable_phases()]
    return dict(status='complete',kind='equilibrium',temperature_K=temperature_K,pressure_Pa=pressure_Pa,
                input_basis='mole_fraction',input_solute_fractions=composition,phases=phases)
