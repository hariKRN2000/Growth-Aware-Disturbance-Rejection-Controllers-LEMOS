# Growth-Aware Optogenetic Control for Cross-Scale Disturbance Rejection in Bacterial Batch Cultures

---

## Repository Structure

```
.
├── model_and_controller_equations/
│   ├── TCS_model_growth_equations.py   # Multiscale ODE (GEAGS) model of the TCS optogenetic circuit 
│   ├── run_constant.py                 # Runs the model under constant light input (open loop)
│   └── run_all_controllers.py          # Implements all controllers (P, PI, PD, PID, aPID, FF, aPID-FF) and the closed-loop simulation loop
│
├── frequency response analysis/
│   └── Sim_TF_Bode_C_as_Disturbance.ipynb   # Frequency response analysis treating cell growth as a disturbance input
│ 
├── robustness analysis/
│   └── robustness_analysis_all_controllers_same_IC.ipynb   # Monte Carlo analysis of all the controllers at different operating regimes
│   └── figures, experimental data, parameters, models, and controller files.
│
├── experiment_data/
│   └── P-FL_OD_run_data_*.csv          # Closed-loop controller experimental data from previous work
│
├── parameters/
│   └── TCS_params_guess_070225.csv     # GEAGS model parameters (kinetic rates and initial conditions)
│
├── sweep_results/
│   └── sweep_results_7_by_7.csv        # Results from a 7×7 parameter sweep
│
├── figures/                            # Pre-generated SVG figures used in the paper
│
├── perturbation_simulation_fixed_gain_PID.ipynb    # Simulates growth perturbations under fixed-gain PID
├── perturbation_simulation_all_controllers.ipynb   # Compares controllers under the same perturbation conditions
├── plot_performance_metrics.ipynb                  # Generates performance metric heatmaps (ITAE)
└── signal_analysis.py                              # Helper functions for computing control performance metrics
```

---

## Installation

```bash
git clone https://github.com/hariKRN2000/Gain-Scheduled-Disturbance-Rejection-Controllers-LEMOS.git
cd Gain-Scheduled-Disturbance-Rejection-Controllers-LEMOS
pip install numpy scipy pandas lmfit matplotlib 
```

Then launch Jupyter and run any of the `.ipynb` notebooks.
