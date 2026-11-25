import json
import numpy as np
from tc_python import TCPython, ThermodynamicQuantity, CalculationException

# Convert temperature from Celsius to Kelvin
temperature_kelvin = 700 + 273.15

# Define the mean and covariance for the normal distribution
mean = [15, 0.5]  # Mean values for Cr (wt-%) and C (wt-%)
covariance = [[10, 0], [0, 0.1]]  # Covariance matrix

# Generate random compositions using a normal distribution
num_samples = 100
random_compositions = np.random.multivariate_normal(mean, covariance, num_samples)

# Initialize results storage
results = []

# Initialize Thermo-Calc Python interface
with TCPython() as session:
    # Set up the system
    system = (session
              .select_database_and_elements("FEDEMO", ["Fe", "C", "Cr"])
              .get_system()
              .with_single_equilibrium_calculation()
              .set_condition(ThermodynamicQuantity.temperature(), temperature_kelvin))  # Temperature in Kelvin
    
    # Loop over random compositions
    for cr_wt, c_wt in random_compositions:
        try:
            # Ensure the compositions are within the specified range
            if 5 <= cr_wt <= 25 and 0 <= c_wt <= 1:
                # Set conditions
                system.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component("Cr"), cr_wt / 100)
                system.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component("C"), c_wt / 100)
                
                # Calculate equilibrium
                result = system.calculate()
                
                # Retrieve volume fraction of phase M7C3
                volume_fraction_M7C3 = result.get_value_of(ThermodynamicQuantity.volume_fraction_of_a_phase("M7C3"))
                
                # Store the result
                results.append({"Cr_wt": cr_wt, "C_wt": c_wt, "Volume_fraction_M7C3": volume_fraction_M7C3})
        
        except CalculationException as e:
            # Handle calculation failure
            print(f"Calculation failed for Cr = {cr_wt} wt-%, C = {c_wt} wt-%: {e}")
            results.append({"Cr_wt": cr_wt, "C_wt": c_wt, "Volume_fraction_M7C3": None})

# Save the results to a JSON file
with open("m7c3_volume_fraction_distribution.json", "w") as json_file:
    json.dump(results, json_file)

# Print the results
for res in results:
    print(res)