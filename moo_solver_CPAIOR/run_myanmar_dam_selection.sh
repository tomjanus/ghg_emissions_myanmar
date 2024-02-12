# Run optimizations with and without loss of area as additional objectives
# Scenario - 1 : 5 objectives: energy, firm_energy, ghg emissions, loss of agricultural land area, loss of forest area
# Scenario - 2 : 3 objectives: energy, firm_energy, ghg emissions

# Run Scenario 1 including built dam statuses 
./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_built.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_built.sol 

# Run Scenario 1 with all dams set to not built
./Myanmar_lp -lp -epsilon 2 -thread 4 -path Basin_Input_Files/mya_5_obj_nobuilt.txt -criteria 5 energy ghg firm_energy loss_agri loss_forest -w 1 2 0.5 0.5 -savename outputs/mya_5_obj_nobuilt.sol 

# Run Scenario 1 including built dam statuses 
./Myanmar_lp -lp -epsilon 0.25 -thread 4 -path Basin_Input_Files/mya_5_obj_built.txt -criteria 3 energy ghg firm_energy -savename outputs/mya_3_obj_built.sol 

# Run Scenario 1 with all dams set to not built
./Myanmar_lp -lp -epsilon 0.25 -thread 4 -path Basin_Input_Files/mya_5_obj_nobuilt.txt -criteria 3 energy ghg firm_energy -savename outputs/mya_3_obj_nobuilt.sol 
