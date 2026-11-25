import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tc_python import *

# Define the alloy composition
composition = {"C": 1.0, "Cr": 10.0}  # in wt-%
database = "FEDEMO"  # Assuming a demo database for Fe-based alloys

# File paths for saving results
json_file_path = "solidification_temperature.json"
png_file_path = "solidification_plot.png"

with TCPython() as session:
    # Set up the system and calculation
    system = (session
              .select_database_and_elements(database, ["Fe", "C", "Cr"])
              .get_system())

    scheil_calculation = (system
                          .with_scheil_calculation()
                          .set_composition_unit(CompositionUnit.MASS_PERCENT))

    for element, wt_pct in composition.items():
        scheil_calculation.set_composition(element, wt_pct)

    # Perform the Scheil calculation
    result = scheil_calculation.calculate()

    # Retrieve temperature and solid fraction data
    temperature, solid_fraction = result.get_values_of(
        ScheilQuantity.temperature(),
        ScheilQuantity.mole_fraction_of_all_solid_phases()
    )

    # Interpolate to find the temperature at 80% solidification
    solid_fraction_curve = interp1d(solid_fraction, temperature)
    temperature_at_80_solid = solid_fraction_curve(0.8)

    # Save the result to a JSON file
    result_data = {"temperature_at_80_solidification_K": temperature_at_80_solid}
    with open(json_file_path, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    # Plot the solid fraction as a function of temperature
    plt.plot(temperature, solid_fraction, label="Solid Fraction")
    plt.axvline(x=temperature_at_80_solid, color='r', linestyle='--', label=f"80% Solidified at {temperature_at_80_solid:.2f} K")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Solid Fraction")
    plt.title("Solidification Curve for Fe-1C-10Cr")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_file_path)
    plt.close()

# Confirm the saved files
print(f"Temperature at 80% solidification saved to {json_file_path} and plot saved to {png_file_path}.")