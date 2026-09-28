
# import required files 

import numpy as np 
import scipy.integrate as scint 
from .TCS_model_growth_equations import run_TCS_model

def run_constant(total_time, initial_conditions, params, constant_input = 'green', growth_perturb = False,
                 perturb_percent=50, perturb_time=600, new_t_final=1800):

    # Simulation parameters
    total_time = total_time  # Total simulation time in minutes
    dt = 1  # Time step in minutes
    initial_conditions = initial_conditions  # Initial mRNA and protein concentrations
    time_dark = 1

    # Variables to store results
    protein_concentrations = [initial_conditions[8]]
    solution_array = np.array([initial_conditions])
    time_array = [0]  # Array to store time points

    # Initial state
    state = initial_conditions
    current_time = 0
    perturbed = False
    


    # Run the simulation
    while current_time < total_time:

        
        # Get the current protein concentration
        protein_concentration = state[8]
        
        # set the input times: 
        if constant_input.lower() == 'green':
            time_green, time_red = 8, 0

        if constant_input.lower() == 'red':
            time_green, time_red = 0, 8

        if constant_input.lower() == 'dark':
            time_green, time_red, time_dark = 0, 0, 5

        
        # Perturbation logic (only applies once)
        if current_time >= perturb_time and growth_perturb and not perturbed:
            restart_ind_percentage = (100 - perturb_percent) / 100
            print("Perturbing now")
            # Find the point where the protein concentration is above 50% of its max
            perturb_time_index = np.argmax(solution_array[:, 9] >= np.max(solution_array[:, 9]) * restart_ind_percentage)
            # Double the total time for the next phase
            total_time = new_t_final
            # Restart from the perturbation point
            state[9] = solution_array[perturb_time_index][9]
            perturbed = True  # Set flag to avoid further perturbations
            print(f"Perturbation applied at time {current_time} to bring to {time_array[perturb_time_index]} minutes")

        
        # Repeat the green-red cycle twice (two 10-minute cycles)
        for _ in range(2):

            # Simulate Dark Period
            t_dark = np.linspace(current_time, current_time + time_dark, int(time_dark/dt) + 100)
            sol = run_TCS_model(state, t_dark, params, 0, input = 'dark')
            state = sol.y[:,-1]
            current_time += time_dark
            time_array = np.concatenate((time_array, sol.t))
        
            # Store protein concentration after dark
            protein_concentrations = np.concatenate((protein_concentrations, sol.y[8,:]))

            # store the solution after dark period: 
            solution_array = np.vstack((solution_array, sol.y.T))

            # Simulate green light period
            if time_green == 0:
                pass
            
            else:
                t_green = np.linspace(current_time, current_time + time_green, int(time_green/dt) + 100)
                sol = run_TCS_model(state, t_green, params, 0, input = 'green')
                state = sol.y[:,-1]
                current_time += time_green
                time_array = np.concatenate((time_array, sol.t))
            
                # Store protein concentration after green light
                protein_concentrations = np.concatenate((protein_concentrations, sol.y[8,:]))

                # store the solution after dark period: 
                solution_array = np.vstack((solution_array, sol.y.T))
            
            # Simulate red light period
            if time_red == 0:
                pass
            
            else:
                t_red = np.linspace(current_time, current_time + time_red, int(time_red/dt) + 100)
                sol = run_TCS_model(state, t_red, params, 0, input = 'red')
                state = sol.y[:,-1]
                current_time += time_red
                time_array = np.concatenate((time_array, sol.t))
                
                # Store protein concentration after red light
                protein_concentrations = np.concatenate((protein_concentrations, sol.y[8,:]))

                # store the solution after dark period: 
                solution_array = np.vstack((solution_array, sol.y.T))

            # Simulate Dark Period
            t_dark = np.linspace(current_time, current_time + time_dark, int(time_dark/dt) + 100)
            sol = run_TCS_model(state, t_dark, params, 0, input = 'dark')
            state = sol.y[:,-1]
            current_time += time_dark
            time_array = np.concatenate((time_array, sol.t))
        
            # Store protein concentration after dark
            protein_concentrations = np.concatenate((protein_concentrations, sol.y[8,:]))

            # store the solution after dark period: 
            solution_array = np.vstack((solution_array, sol.y.T))
        

        
            
    return time_array, protein_concentrations, solution_array