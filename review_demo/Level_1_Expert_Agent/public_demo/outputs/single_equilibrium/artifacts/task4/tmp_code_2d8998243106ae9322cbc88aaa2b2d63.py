import json
from tc_python import TCPython, ThermodynamicQuantity

# Convert temperature from Celsius to Kelvin
temperature_kelvin = 700 + 273.15

# Initialize Thermo-Calc Python interface
with TCPython() as session:
    # Set up the system
    system = (session
              .select_database_and_elements("FEDEMO", ["Fe", "C", "Cr"])
              .get_system()
              .with_single_equilibrium_calculation()
              .set_condition(ThermodynamicQuantity.temperature(), temperature_kelvin)  # Temperature in Kelvin
              .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("C"), 0.01)  # Mole fraction of Carbon
              .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Cr"), 0.1))  # Mole fraction of Chromium
    
    # Calculate equilibrium
    result = system.calculate()
    
    # Retrieve volume fraction of phase M23C6
    volume_fraction_M23C6 = result.get_value_of(ThermodynamicQuantity.volume_fraction_of_a_phase("M23C6"))
    
    # Print the result
    print(f"Volume fraction of phase M23C6: {volume_fraction_M23C6}")
    
    # Save the result to a JSON file
    output_data = {
        "Temperature_C": 700,
        "X(C)": 0.01,
        "X(Cr)": 0.1,
        "Volume_fraction_M23C6": volume_fraction_M23C6
    }
    
    with open("fe_c_cr_volume_fraction_M23C6.json", "w") as json_file:
        json.dump(output_data, json_file)