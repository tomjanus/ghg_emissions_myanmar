# Collection of notebooks for post-processing GHG emission results obtained from [RE-Emission](https://github.com/tomjanus/reemission)
### Tomasz Janus
### University of Manchester
### email: tomasz.janus@manchester.ac.uk; tomasz.k.janus@gmail.com

## The list of notebooks in the order of execution

* **Notebook_1_run_batch_simulation.ipynb** - This Jupyter notebook in **Python** runs a series of simulations with RE-Emission - a tool for estimating GHG emissions from reservoirs. It creates a number of RE-Emission input files in `inputs/reemission`, performs calculations for each input file and saves the outputs to `outputs/reemission`. The output files are saved in `.xlsx` and `.json` formats. The input data for RE-Emission is stored in `inputs/reemission/reemission_inputs.json`. The inputs are obtained from the GEOspatial CAtchment and REservoir analysis Tool [GEOCaret](https://github.com/Reservoir-Research/geocaret).

* **Notebook_2_process_hydropower.Rmd** - This notebook written in **RMarkDown** compares hydropower (HP) production-related figures in the Myanmar's existing and future hydroelectric power plants provided in the [IFC database of dams](https://www.ifc.org/en/insights-reports/2018/sea-of-the-hydropower-sector-in-myanmar-resources-page) against the simulated values from the national water resources model in [Pywr](https://github.com/pywr/pywr). The map of the water resources model, the delineated reservoirs and emission estimates can be found [here](https://tomjanus.github.io/mya_emissions_map/). *FYI* - The model on the map may not be 100% accurate, especially with regards to reservoir parameters, as it has been recently undegrgoing changes independently of this work and these changes may not be reflected in the visualisations. However, the topology and most of the parameter values should nevertheless be correct. The comparison figures produced in the notebook are saved to `figures/ifc_pywr_power_comparison`. The merged data from the IFC database and the simulations is saved to `intermediate/merged_table.xlsx`.

* **Notebook_3_combine_outputs.ipynb** - This Jupyter notebook written in **R** combines multiple files in `outputs/reemission` containing outputs from batch RE-Emission simulations into a single 'combined' tabular dataset and saves this combineds data in `.csv` and `.xlsx` formats. Both files, i.e. `combined_outputs.csv` and `combined_outputs.xlsx`, are saved to the folder `outputs/reemission/combined/`.

* **Notebook_4_ghg_emission_plots.ipynb** - This Jupyter notebook in **Python** creates statistical plots with emission outputs. The plots summarize net (aerial) emissions with different values of categorical variables, such as landuse intensity (low/high), soil type (mineral/organic) and water intake depth (shallow/deep), and visualise the distributions of emissions with all reservoirs types - hydroelectric, irrigation and multipurpose. The output figures are saved to `figures/ghg_visualisation`.

* **Notebook_5_create_reservoir_tile_maps.Rmd** - This notebook written in **RMarkDown** creates tile plots of all delineated reservoirs. The tile plots of reservoir contours are saved to `figures/maps/`.

* **Notebook_6_create_reservoir_maps.ipynb** - This notebook in **R** creates maps showing emissions and emission intensities of the reservoirs in Myanmar. The plots are saved to `figures/maps`.

* **Notebook_7_run_catboost_lightgbm_xgboost_regressions.ipynb** - This notebook in **Python** visualises re-emission input data and explores the data structure, checks feature scores for CO2 and CH4 regression using conventional statistical methods, then fits the CO2 and CH4 estimates from re-emission using boosted tree regression models using pre-selected re-emission input data. Three boosted tree models are used: XGBoost, CATBoost and LightGBM. The regression model are then used to provide explanation about model predictions on a model-level as well as on the instance-level. The notebook saves figures to `figures/data_exploration` and `figures/model_explanation`. The fitted models are saved to `outputs/model_explanations`.

* **Notebook_8_dim_red_and_clustering_of_feature_importances.ipynb** - This notebook in **Python** clusters reservoir based on their similarities in the feature importance space. The feature importance space is first reduced using various dimensionality reduction methods. The notebook implements different clustering as well as dimensionality reduction algorithms. The figures are saved to `figures/clustering`. Cluster data used for mapping with `Notebook_9_create_clustering_maps.ipynb` in R are saved to `intermediate/density_mapping`.

* **Notebook_9_create_clustering_maps.ipynb** - This notebook in **R** plots three types of maps and saves them to `figures/maps`. The first map is emission intensity with a digital elevation map in the background. The two other types of maps are cluster maps based on cluster on features and feature ranks and coloured using voronoi polygons. 

* **Notebook_9b_process_additional_information_from_water_resource_models.ipynb** - This Jupyter notebook in **Python** processes outputs from the water resources models containing reservoir levels, turbine flows and recorded spill flows and creates input data for designing a ML learning model and subsequent explanations using xAI.

* **Notebook_9c_generate_breakdowns_for_selected_reservoirs_for_figure3.ipynb** - This Jupyter notebook in **Python** trains a boosted tree model predicting emission intensity of HP reservoirs and use DALEX to interpret the results. The breakdown plots for selected reservoirs are plotted and saved to figures in `.svg` format.

* **Notebook_10_generate_inputs_for_MOO.ipynb** - This notebook written in **Python** creates dataframes required for preparing inputs to multiobjective optimisation (MOO) study. The dataframes are saved as a collection of `.csv` files.

* **Notebook_11_create_input_files_for_MOO.ipynb** - This notebook written in **Python** prepares input `.txt` files for the MOO algorithm.

* **Notebook_12_run_MOO_and_visualise.ipynb** - This notebook written in **Python** runs and post-processes optimal dam portfolio selection study in Myanmar.

* **Notebook_13_plot_maps_of_MOO_results.ipynb** - This notebook written in **Python** visualises the results of the optimization study on a composite figure consisting of 6 maps showing the selected dams for 6 different optimization scenarios indicated on the Pareto front plots generated in `Notebook_12`. The figure additionally features a number of statistical plots showing distributions of dams with respect to HP generation and elevation across all solutions. The objectives for each solutions are plotted in radar plots.

* **calculate_statistical_figures_for_the_manuscript.ipynb** - This short notebook is used for calculating statistical numbers for reporting in the manuscript either directly in text or in tables.



