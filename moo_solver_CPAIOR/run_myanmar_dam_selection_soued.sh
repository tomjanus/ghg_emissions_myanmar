# Run optimizations with and without loss of area as additional objectives
# Scenario - 1 : 5 objectives: energy, firm_energy, ghg emissions, loss of agricultural land area, loss of forest area
# Scenario - 2 : 3 objectives: energy, firm_energy, ghg emissions

# RESTRICT THE ANALYSIS FOR 5 OBJECTIVES ONLY BECAUSE THE FINAL MANUSCRIPT PRESENTS THE RESULTS WITH 5 OBJECTIVES
# I.E. SCENARIO 2 WITH 3 OBJECTIVES, WHICH WAS RUN FIRST USING GHG EMISSIONS CALCULATED FROM RE-EMISSION
# WITH G-RES METHODOLOGY IS NOT REPEATED HERE FOR EMISSIONS CALCULATED FROM EMISSION FACTORS OF SOUED ET AL.

# Run Scenario 1 including built dam statuses 
./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_built_soued.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_built_soued.sol 

# Run Scenario 1 with all dams set to not built
./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_nobuilt_soued.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_nobuilt_soued.sol


