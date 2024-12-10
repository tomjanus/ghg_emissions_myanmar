#!/bin/bash

# Run optimizations with and without loss of area as additional objectives
# Scenario - 1 : 5 objectives: energy, firm_energy, ghg emissions, loss of agricultural land area, loss of forest area
# Scenario - 2 : 3 objectives: energy, firm_energy, ghg emissions

# RESTRICT THE ANALYSIS FOR 5 OBJECTIVES ONLY BECAUSE THE FINAL MANUSCRIPT PRESENTS THE RESULTS WITH 5 OBJECTIVES
# I.E. SCENARIO 2 WITH 3 OBJECTIVES, WHICH WAS RUN FIRST USING GHG EMISSIONS CALCULATED FROM RE-EMISSION
# WITH G-RES METHODOLOGY IS NOT REPEATED HERE FOR EMISSIONS CALCULATED FROM EMISSION FACTORS OF SOUED ET AL.

# Function to display help message
usage() {
    echo "Usage: $0 [-3] [-5] [-a]"
    echo "Options:"
    echo "  -3    Run only 3-criteria optimizations"
    echo "  -5    Run only 5-criteria optimizations"
    echo "  -a    Run all optimizations (default if no options are provided)"
    exit 1
}

# Check if no arguments are provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize flags
run_3=false
run_5=false

# Parse command-line options
while getopts "35a" opt; do
    case ${opt} in
        3) run_3=true ;;
        5) run_5=true ;;
        a) run_3=true; run_5=true ;;
        *) usage ;;
    esac
done

# Default behavior: Run all optimizations if no options are provided
if ! $run_3 && ! $run_5; then
    run_3=true
    run_5=true
fi

# Run 5-criteria optimizations
if $run_5; then
    echo "Running 5-criteria optimizations..."
    # Run Scenario 1 including built dam statuses 
    ./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_built_soued.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_built_soued.sol 
    # Run Scenario 1 with all dams set to not built
    ./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_nobuilt_soued.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_nobuilt_soued.sol
fi

# Run 3-criteria optimizations
if $run_3; then
    echo "Running 3-criteria optimizations..."
    ./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_3_obj_built_soued.txt -criteria 3 energy ghg firm_energy -savename outputs/mya_3_obj_built_soued.sol 
    # Run Scenario 1 with all dams set to not built
    ./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_3_obj_nobuilt_soued.txt -criteria 3 energy ghg firm_energy -savename outputs/mya_3_obj_nobuilt_soued.sol

