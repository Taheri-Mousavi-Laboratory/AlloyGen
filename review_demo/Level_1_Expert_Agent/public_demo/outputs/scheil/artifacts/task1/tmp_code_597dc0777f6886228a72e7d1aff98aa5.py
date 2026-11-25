import json
import numpy as np
import matplotlib.pyplot as plt
from tc_python import *

# Define the alloy composition
composition = {"C": 1.0, "Cr": 10.0}  # in wt-%
database = "FEDEMO"  # Assuming a demo database for Fe-based alloys

# File paths for saving results
json_file_path = "scheil_results.json"
png_file_path = "latent_heat_plot.png"

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

    # Retrieve temperature and latent heat data
    temperature, latent_heat = result.get_values_of(
        ScheilQuantity.temperature(),
        ScheilQuantity.latent_heat_per_gram()
    )

    # Save results to a JSON file
    results_data = {"temperature_K": temperature, "latent_heat_J_per_g": latent_heat}
    with open(json_file_path, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    # Plot the latent heat as a function of temperature
    plt.plot(temperature, latent_heat, label="Latent Heat")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Latent Heat (J/g)")
    plt.title("Latent Heat of Solidification for Fe-1C-10Cr")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_file_path)
    plt.close()

# Confirm the saved files
print(f"Results saved to {json_file_path} and plot saved to {png_file_path}.")