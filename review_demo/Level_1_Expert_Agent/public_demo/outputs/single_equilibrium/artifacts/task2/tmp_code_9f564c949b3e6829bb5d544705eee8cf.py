import json
from tc_python import TCPython, LoggingPolicy

# Initialize Thermo-Calc Python interface
with TCPython(logging_policy=LoggingPolicy.FILE, log_file="solver.log") as session:
    # Set up the system
    system = session.get_system(database_name="TCFE", elements=["Fe", "C", "Cr"])
    
    # Define conditions
    system.set_condition("T", 2000)  # Temperature in Kelvin
    system.set_condition("X(C)", 0.01)  # Mole fraction of Carbon
    system.set_condition("X(Cr)", 0.1)  # Mole fraction of Chromium
    
    # Calculate equilibrium
    result = system.calculate_equilibrium()
    
    # Retrieve Gibbs energy
    gibbs_energy = result.get_value_of("G")  # Gibbs energy in J/mol
    
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