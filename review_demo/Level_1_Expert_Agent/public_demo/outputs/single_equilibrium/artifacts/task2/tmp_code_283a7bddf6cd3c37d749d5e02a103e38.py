import json
from tc_python import TCPython, ThermodynamicQuantity

# Initialize Thermo-Calc Python interface
with TCPython() as session:
    # Set up the system
    system = (session
              .select_database_and_elements("FEDEMO", ["Fe", "C", "Cr"])
              .get_system()
              .with_single_equilibrium_calculation()
              .set_condition(ThermodynamicQuantity.temperature(), 2000)  # Temperature in Kelvin
              .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("C"), 0.01)  # Mole fraction of Carbon
              .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component("Cr"), 0.1))  # Mole fraction of Chromium
    
    # Calculate equilibrium
    result = system.calculate()
    
    # Retrieve Gibbs energy of the entire system
    gibbs_energy = result.get_value_of(ThermodynamicQuantity.gibbs_energy())  # Gibbs energy in J/mol
    
    # Print the result
    print(f"Gibbs energy of the system: {gibbs_energy} J/mol")
    
    # Save the result to a JSON file
    output_data = {
        "Temperature": 2000,
        "X(C)": 0.01,
        "X(Cr)": 0.1,
        "Gibbs_energy": gibbs_energy
    }
    
    with open("fe_c_cr_gibbs_energy.json", "w") as json_file:
        json.dump(output_data, json_file)