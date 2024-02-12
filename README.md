# Collection of notebooks for post-processing GHG emission results

## List of notebooks in the order of execution

* **Notebook_1_run_batch_simulation.ipynb** - This notebook in **Python** runs a series of simulations with RE-Emission. It creates a number of reemission input files in `inputs/reemission` and performs calculations for each reemission file and saves the outputs in `outputs/reemission`. The reemission output files are saved in `.xlsx` and `.json` formats. The "base" input data for reemission is in `inputs/reemission/reemission_inputs.json` file.

* **Notebook_2_prcess_hydropower.Rmd** - This notebook written in **RMarkDown** compares HP production figures from IFC database with the values obtained from our own Myanmar's' water resources model created in Pywr. The comparison figures are saved to `figures/ifc_pywr_power_comparison`. The merged data from IFC and the outputs from the water resources model are saved to an Excel file in `intermediate/merged_table.xlsx`.

* **Notebook_3_combine_outputs.ipynb** - This short notebook in **R** combines outputs from batch RE-Emission simulation that are stored as multiple files in `outputs/reemission` into a single file using two formats: `.csv` and `.xlsx`. Both files (`combined_outputs.csv` and `combined_outputs.xlsx`) are saved in `outputs/reemission/combined/` directory.

* **Notebook_4_ghg_emission_plots.ipynb** - This notebook in **Python** creates statistical plots summarizing the emission outputs. The plots summarise net emissions with low and high landuse intensity on mineral and organic soils, with shallow and deep intakes and show the distributions of emissions among all analysed hydroelectric, irrigation and multipurpose reservoirs. The output figures are saved to `figures/ghg_visualisation`.

* **Notebook_5_create_reservoir_tile_maps.Rmd** - This notebook written in **RMarkDown** creates tile plots of all delineated reservoirs. The tile plots of reservoir contours are saved to `figures/maps/`.

* **Notebook_6_create_reservoir_maps.ipynb** - This notebook in **R** creates maps showing emissions and emission intensities in Burmese reservoirs. The plots are saved to `figures/maps`.

* **Notebook_7_run_catboost_lightgbm_xgboost_regressions.ipynb** - This notebook in **Python** visualises re-emission input data and explores the data structure, checks feature scores for CO2 and CH4 regression using conventional statistical methods, then fits the CO2 and CH4 estimates from re-emission using boosted tree regression models using pre-selected re-emission input data. Three boosted tree models are used: XGBoost, CATBoost and LightGBM. The regression model are then used to provide explanation about model predictions on a model-level as well as on the instance-level. The notebook saves figures to `figures/data_exploration` and `figures/model_explanation`. The fitted models are saved to `outputs/model_explanations`.

* **Notebook_8_dim_red_and_clustering_of_feature_importances.ipynb** - This notebook in **Python** clusters reservoir based on their similarities in the feature importance space. The feature importance space is first reduced using various dimensionality reduction methods. The notebook implements different clustering as well as dimensionality reduction algorithms. The figures are saved to `figures/clustering`. Cluster data used for mapping with `Notebook_9_create_clustering_maps.ipynb` in R are saved to `intermediate/density_mapping`.

* **Notebook_9_create_clustering_maps.ipynb** - This notebook in **R** plots three types of maps and saves them to `figures/maps`. The first map is emission intensity with a digital elevation map in the background. The two other types of maps are cluster maps based on cluster on features and feature ranks and coloured using voronoi polygons. 

* **Notebook_10_generate_inputs_for_MOO.ipynb** - This notebook written in **Python** creates dataframes required for preparing inputs to multiobjective optimisation (MOO) study. The dataframes are saved as a collection of `.csv` files.

* **Notebook_11_create_input_files_for_MOO.ipynb** - This notebook written in **Python** prepares input `.txt` files for the MOO algorithm.

* **Notebook_12_run_MOO_and_visualise.ipynb** - This notebook written in **Python** runs and post-processes optimal dam portfolio selection study in Myanmar.




