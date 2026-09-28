import numpy as np
from scipy.optimize import minimize
import time
from collections import deque
from .TCS_model_growth_equations import run_TCS_model


# ------------------- Utilities -------------------
def clamp_time_green(time_green, control_interval=8):
    time_green = max(0, min(time_green, control_interval))
    time_red = control_interval - time_green
    time_green, time_red = zero_if_close(time_green), zero_if_close(time_red)
    return time_green, time_red

def zero_if_close(x, tol=1e-4):
    """Need to define this function to clamp the values to 0.
       Sometimes optimizers will return values that are too close to 0 but not quite 0."""
    return 0.0 if abs(x) < tol else x


# ------------------ Controller Definitions ------------------

def p_controller(protein, set_point, gain):
    error = set_point - protein
    time_green = gain * error
    return clamp_time_green(time_green)

def pi_controller(protein, set_point, gain, error_array, error_time_array, tau_I, integral_error, mode='model'):
    error = set_point - protein
    p_term = error

    if mode == 'discrete':
        # Trapezoidal integration
        integral = 0.0
        if len(error_array) >= 2 and len(error_time_array) >= 2:
            for i in range(1, len(error_array)):
                dt = error_time_array[i] - error_time_array[i - 1]
                avg_error = 0.5 * (error_array[i] + error_array[i - 1])
                integral += avg_error * dt
    elif mode == 'model':
        integral = integral_error
    else:
        raise ValueError(f"Invalid mode for integral estimation: {mode}")

    i_term = (1 / tau_I) * integral
    time_green = gain * (p_term + i_term)
    return clamp_time_green(time_green)

def pd_controller(protein, set_point, gain, error_array, time_array, tau_D):
    error = set_point - protein
    p_term = error

    if len(error_array) > 1:
        dt = time_array[-2] - time_array[-1]
        d_error = (error_array[-2] - error_array[-1]) / dt if dt != 0 else 0
    else:
        d_error = 0

    d_term = tau_D * d_error
    time_green = gain * (p_term + d_term)
    return clamp_time_green(time_green)

def pid_controller(protein, set_point, gain, error_array, time_array, tau_I, tau_D, integral_error, mode='model'):
    error = set_point - protein
    p_term = error

    # Integral Term
    integral = integral_error
    i_term = (1 / tau_I) * integral

    # Derivative Term
    if len(error_array) > 1:
        dt = time_array[-1] - time_array[-2]
        d_error = (error_array[-1] - error_array[-2]) / dt if dt != 0 else 0
    else:
        d_error = 0
    d_term = tau_D * d_error

    time_green = gain * (p_term + i_term + d_term)
    return clamp_time_green(time_green)

def pid_controller_growth_aware(protein, set_point, gain, error_array,
                                 time_array, tau_I, tau_D, integral_error,f, gain_0=0.005, tau_D_0=10,
                                 controller_mode="PID", perturb_percent=50, f_perturbed=0.8, FF_gain=1):
    """This function is used to simulate the gain-scheduled PID controller. Takes in the growth factor f as the input 
    and computes the gains for PID controller based on the growth rate."""

    error = set_point - protein
    
    if controller_mode == "PID": 

        p_term = error

        # Integral Term
        integral = integral_error
        i_term = (1 / tau_I) * integral

        # Derivative Term
        if len(error_array) > 1:
            dt = time_array[-1] - time_array[-2]
            d_error = (error_array[-1] - error_array[-2]) / dt if dt != 0 else 0
        else:
            d_error = 0
        
        d_term = tau_D * d_error

        time_green = gain * (p_term + i_term + d_term)

    elif controller_mode == "PID-GS": # to switch to gain-scheduled PID

        # Proportional Term
        p_term = error

        # Integral Term
        integral = integral_error

        i_term = (1 / tau_I) * integral

        # Derivative Term
        if len(error_array) > 1:
            dt = time_array[-1] - time_array[-2]
            d_error = (error_array[-1] - error_array[-2]) / dt if dt != 0 else 0
        else:
            d_error = 0
        
        # Adjust the derivative time constant as function of growth
        tau_D = tau_D * f + tau_D_0
        d_term = tau_D * d_error 

        # Scaling Kc as a function of growth
        gain = gain * f + gain_0

        time_green = gain * (p_term + i_term + d_term)


    elif controller_mode == "FF": # switch to plain Feed Forward
        
        disturbance_extent = f_perturbed - f

        time_green = disturbance_extent * FF_gain

    elif controller_mode == "PID-FF": # switch to combined Feed Forward and PID
        
        # PID input
        # Proportional Term
        p_term = error

        # Integral Term
        integral = integral_error

        i_term = (1 / tau_I) * integral

        # Derivative Term
        if len(error_array) > 1:
            dt = time_array[-1] - time_array[-2]
            d_error = (error_array[-1] - error_array[-2]) / dt if dt != 0 else 0
        else:
            d_error = 0
        
        # derivative time constant
        d_term = tau_D * d_error 

        time_green_PID = gain * (p_term + i_term + d_term)
        
        # Feedforward input
        disturbance_extent = f_perturbed - f

        time_green_FF = disturbance_extent * FF_gain

        time_green = time_green_PID + time_green_FF

    elif controller_mode == "PID-GS-FF": # switch to combined PID-GS and Feed Forward
        
        # PID input
        # Proportional Term
        p_term = error

        # Integral Term
        integral = integral_error

        i_term = (1 / tau_I) * integral

        # Derivative Term
        if len(error_array) > 1:
            dt = time_array[-1] - time_array[-2]
            d_error = (error_array[-1] - error_array[-2]) / dt if dt != 0 else 0
        else:
            d_error = 0
        
        # Adjust the derivative time constant as function of growth
        tau_D = tau_D * f + tau_D_0
        d_term = tau_D * d_error 

        # Scaling Kc as a function of growth
        gain = gain * f + gain_0

        time_green_PID = gain * (p_term + i_term + d_term)
        
        # Feedforward input
        disturbance_extent = f_perturbed - f

        time_green_FF = disturbance_extent * FF_gain

        time_green = time_green_PID + time_green_FF



    return clamp_time_green(time_green)



# ------------------ Main Control Runner ------------------

def run_control(controller_type, total_time, set_point, initial_conditions, params, gain,
                tau_I=1500, tau_D=120, integral_mode='model', perturb_percent=50, perturb_time=600,
                new_t_final=1800, C_max=713694117.0, initial_controller="PID", switch_to_controller="PID", FF_gain=1,
                print_perturb_message=False, params_after_perturb=None):

 
    time_dark = 1
    state = np.asarray(initial_conditions, dtype=float).copy()
    current_time = 0

    # Plant parameters used by the ODE model.
    # Start every trajectory with the same nominal parameters.
    active_params = params

    protein_concentrations = [state[8]]
    solution_array = np.array([state])
    time_array = [0]
    time_green_values = [4]

    error_array = [0]
    error_time_array = [0]
    opt_times = []
    perturbed = False # set it as False before any perturbation happens
    C_measured = [0.0]
    t_measured = [0.0]


    f_perturbed = 0.6 # This is just a placeholder, which is later modified inside the loop when the perturbation occurs.

    while current_time < total_time:
        protein = state[8]
        error = set_point - protein
        

        # --- Controller Selection ---
        if controller_type == "P":
            time_green, time_red = p_controller(protein, set_point, gain)

        elif controller_type == "PD":
            time_green, time_red = pd_controller(protein, set_point, gain, error_array, error_time_array, tau_D)

        elif controller_type == "PI":
            time_green, time_red = pi_controller(
                protein, set_point, gain, error_array, error_time_array, tau_I, state[10], mode=integral_mode)
            
        elif controller_type == "PID":
            time_green, time_red = pid_controller(
                protein, set_point, gain, error_array, error_time_array, tau_I, tau_D, state[10], mode=integral_mode)

        elif controller_type == "PID-growth-perturb":

            restart_ind_percentage = (100 - perturb_percent)/100 

            if current_time >= perturb_time:
                if perturbed == False:

                    if print_perturb_message:
                        print(f"Perturbing now, applying the {controller_type} algorithm")

                    # Store the value of f when perturbed: 
                    f_perturbed = solution_array[-1, 9]/C_max

                    # Find the point where the protein concentration is above 50% of its max
                    perturb_time_index = np.argmax(solution_array[:, 9] >= np.max(solution_array[:, 9]) * restart_ind_percentage)
                    
                    # Update the total time to the new user-defined experiment end time
                    total_time = new_t_final

                    # Reset the appropriate states to the perturbation point
                    state[9] = solution_array[perturb_time_index][9]
                    state[10] = 0 # Integral windup reset

                    # Introduce plant uncertainty only when the disturbance occurs.
                    if params_after_perturb is not None:
                        active_params = params_after_perturb
                    
                    perturbed = True  # Set flag to avoid further perturbations

                    if print_perturb_message:
                        print(f"Perturbation applied at time {current_time} to bring to {time_array[perturb_time_index]} minutes")

            start_time = time.time()
            f_current = solution_array[-1, 9]/C_max

            if perturbed:
                mode = switch_to_controller
            
            else:
                mode = initial_controller
                
            time_green, time_red = pid_controller_growth_aware(
                protein, set_point, gain, error_array, time_array, tau_I, tau_D, state[10], f_current,
                controller_mode=mode, perturb_percent=perturb_percent, f_perturbed=f_perturbed, FF_gain=FF_gain)
            end_time = time.time()
            elapsed_time = end_time - start_time
            opt_times.append(elapsed_time)

        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        time_green_values.append(time_green)

        # --- Light Cycle: Dark -> Green -> Red -> Dark (x2) ---
        for _ in range(2):
            for phase, duration in [('dark', time_dark), ('green', time_green), ('red', time_red), ('dark', time_dark)]:
                if duration > 0:
                    t_phase = np.linspace(current_time, current_time + duration, 100)
                    sol = run_TCS_model(state, t_phase, active_params, set_point, input=phase)
                    state = sol.y[:, -1]
                    current_time += duration
                    time_array = np.concatenate((time_array, sol.t))
                    protein_concentrations = np.concatenate((protein_concentrations, sol.y[8, :]))
                    solution_array = np.vstack((solution_array, sol.y.T))
                    error_array = np.concatenate((error_array, set_point - sol.y[8,:]))
                    error_time_array = np.concatenate((time_array, sol.t))

            C_measured.append(solution_array[-1,9])
            t_measured.append(time_array[-1])

    
    return time_array, protein_concentrations, solution_array, time_green_values, opt_times