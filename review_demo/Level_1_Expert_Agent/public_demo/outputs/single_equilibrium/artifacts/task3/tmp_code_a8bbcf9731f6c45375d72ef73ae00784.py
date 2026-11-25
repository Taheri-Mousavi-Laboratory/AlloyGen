import json
import numpy as np
import matplotlib.pyplot as plt
from tc_python import TCPython, ThermodynamicQuantity, CompositionUnit, CalculationException

# Convert temperature from Celsius to Kelvin
temperature_kelvin = 700 + 273.15

# Define the range of Chromium compositions in wt-%
cr_compositions_wt = np.linspace(5, 25, num=21)

# Initialize results storage
results = []

# Initialize Thermo-Calc Python interface
with TCPython() as session:
    # Set up the system
    system = (session
              .select_database_and_elements("FEDEMO", ["Fe", "C", "Cr"])
              .get_system()
              .with_single_equilibrium_calculation()
              .set_condition(ThermodynamicQuantity.temperature(), temperature_kelvin)  # Temperature in Kelvin
              .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("C"), 0.01))  # Mole fraction of Carbon
    
    # Loop over Chromium compositions
    for cr_wt in cr_compositions_wt:
        try:
            # Convert Cr composition from wt-% to mole fraction
            system.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component("Cr"), cr_wt / 100)
            
            # Calculate equilibrium
            result = system.calculate()
            
            # Retrieve volume fraction of phase M23C6
            volume_fraction_M23C6 = result.get_value_of(ThermodynamicQuantity.volume_fraction_of_a_phase("M23C6"))
            
            # Store the result
            results.append({"Cr_wt": cr_wt, "Volume_fraction_M23C6": volume_fraction_M23C6})
        
        except CalculationException as e:
            # Handle calculation failure
            print(f"Calculation failed for Cr = {cr_wt} wt-%: {e}")
            results.append({"Cr_wt": cr_wt, "Volume_fraction_M23C6": None})

# Plot the results
cr_values = [res["Cr_wt"] for res in results if res["Volume_fraction_M23C6"] is not None]
m23c6_values = [res["Volume_fraction_M23C6"] for res in results if res["Volume_fraction_M23C6"] is not None]

plt.plot(cr_values, m23c6_values, marker='o')
plt.xlabel("Cr composition (wt-%)")
plt.ylabel("Volume fraction of M23C6")
plt.title("Phase fraction of M23C6 vs. Cr composition")
plt.grid(True)
plt.savefig("m23c6_phase_fraction_vs_cr.png")
plt.show()

# Save the results to a JSON file
with open("m23c6_phase_fraction_results.json", "w") as json_file:
    json.dump(results, json_file)