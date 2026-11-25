# TC-Python 2024a: Scheil quantities

`ScheilQuantity` methods:

| ScheilQuantity method | Quantity and units |
| --- | --- |
| `temperature()` | Temperature, K |
| `latent_heat_per_gram()` | Cumulative latent heat per mass, J/g |
| `latent_heat_per_mole()` | Cumulative latent heat per amount, J/mol |
| `mole_fraction_of_all_solid_phases()` | Total molar fraction of solids |
| `mass_fraction_of_all_solid_phases()` | Total mass fraction of solids |
| `volume_fraction_of_all_solid_phases()` | Total volume fraction of solids |

Latent heat accumulates phase-transformation heat from liquidus to the current
temperature, excluding sensible cooling.

`ScheilCalculationResult.get_values_of(x_quantity, y_quantity)` returns
`(x_values, y_values)` as Python lists sorted by x.

Source: [TC-Python 2024a guide](https://thermocalc.com/wp-content/uploads/Documentation/Archive/2020-2024/2024a/2024a-tc-python-api-programmer-guide.pdf),
printed pp. 94, 304–306.
