"""Read small physical diagnostics from this run's existing scientific records.

No solver calls, file search, selection, or changes to indicator definitions.
"""
import json
from pathlib import Path
from scientific_calculator import coarsening
from scientific_tools.tools import HERE as SCIENCE, REPO


def describe(row, run_folder):
    root = Path(run_folder).resolve()
    output = dict(status='available', equilibrium=[], solidification=[])
    for record in row.get('records', []):
        path = (REPO / record).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Diagnostic records must belong to the current run')
        raw = json.loads(path.read_text())
        if raw.get('status') != 'complete':
            continue
        temperature = raw.get('temperature_K')
        if 'phases' in raw and temperature is not None:
            if not any(abs(temperature - target) < 1e-6 for target in (523.15, 933.15)):
                continue
            phases = [dict(name=p['name'], mol_percent=100*p['mole_fraction'],
                           composition_at_percent={e:100*v for e,v in p['composition_mole_fraction'].items()},
                           Er_Zr_bulk_mole_inventory={e:p['mole_fraction']*p['composition_mole_fraction'][e]
                                                      for e in ('Er','Zr')})
                      for p in raw['phases'] if p['mole_fraction'] > 1e-12]
            item = dict(temperature_C=temperature-273.15, phases=phases, source=record)
            if 'J' in row and abs(temperature-523.15) < 1e-6:
                params = json.loads((SCIENCE/'settings.json').read_text())['parameters']
                d = coarsening(raw['phases'], params)
                item['coarsening_components'] = {k:v for k,v in d.items()
                    if k not in ('m_historical_scaled', 'L12_mole_fraction')}
                item['parameter_scope'] = 'Same fixed historical diffusivity/lattice parameters as J; descriptive components, not extra optimization objectives.'
            output['equilibrium'].append(item)
        curves = raw.get('curves', {})
        if 'solid_mole_fraction' in curves:
            total = curves['solid_mole_fraction']
            indices = sorted(range(len(total['values'])), key=lambda i:-total['temperature_K'][i])
            index = min(indices, key=lambda i:abs(total['values'][i]-.02))
            start_T = total['temperature_K'][index]
            amounts = {}
            for name, curve in curves.items():
                if not name.startswith('phase_'):
                    continue
                if curve['temperature_K'] != total['temperature_K']:
                    raise ValueError('Phase histories are not aligned with the saved FR history')
                if curve['values'][index] > 1e-12:
                    amounts[name[6:]] = 100*curve['values'][index]
            output['solidification'].append(dict(
                FR_start_temperature_C=start_T-273.15,
                FR_start_total_solid_mol_percent=100*total['values'][index],
                accumulated_phases_at_FR_start_mol_percent=amounts,
                terminal_temperature_C=total['temperature_K'][indices[-1]]-273.15,
                convention='Existing nearest measured 2% total-solid cutoff; no interpolation or metric change.', source=record))
    if not output['equilibrium'] and not output['solidification']:
        output['status'] = 'unavailable_in_saved_records'
    return output
