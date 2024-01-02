"""Functions for running kriging interpolation """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
import skgstat as skg
import json

def convert_2_json(geodataframe):
    """
    Convert to GeoDataFrame features to rasterio friendly format
    Input is a GeoDataFrame
    """
    return [json.loads(geodataframe.to_json())["features"][0]["geometry"]]


def rmse(residuals):
    return np.sqrt(np.sum((residuals)**2)/len(residuals))


def loocv_kriging(x, y, z, model = "exponential", direction = False, azimuth = 0, tolerance = 22.5):
    """
    Function for ordinary kriging combined with a Leave-one-out cross validation (LOOCV) approach
    Function iterates through each point of x,y,z Dataframe, each time leaving one point out. 
    loocv_kriging will return the interpolated values and the residuals as arrays.
    An ordinary kriging based on specified VARIOGRAM will be performed for each point in iteration.
    Difference will be calculated
    INPUTS:
    x, y  
            arrays containing coordinates of point data
    z 
            array containing data values that are of interest for interpolation
    model
            Variogram model to be used, e.g. "exponential" (default), "spherical", "gaussian"
            For more information, see documentation of scikit-gstat
            https://scikit-gstat.readthedocs.io/en/latest/userguide/variogram.html#variogram-models
    direction 
            wether to use a Directional Variogram 
            https://scikit-gstat.readthedocs.io/en/latest/reference/directionalvariogram.html#skgstat.DirectionalVariogram
            default= False, 
            if True remember to set azimuth and tolerance!
    azimuth
            default = 0
    tolerance
            default = 22.5
    """
    from skgstat import OrdinaryKriging
    ## Initialise two empty vectors to be filled with in a loop
    residuals = []
    rainfall_krig = []
    
    # combine the data array in one structe
    kriging_data = pd.DataFrame({"x_val":x , 
                             "y_val":y, 
                             "rainfall":z})
    
    for index, row in kriging_data.iterrows():
        # drop one point from dataset
        loocv = kriging_data.drop([index])
        # from coords and values for variogram
        loocv_coords = list(zip(loocv.x_val, loocv.y_val))
        loocv_values = loocv.rainfall
        # set up variogram model
        if direction == False:
            V = skg.Variogram(loocv_coords, loocv_values,
                              model = model,
                              # maxlag= 300000,
                              fit_range = 90000,
                              fit_sill = 40000,
                              fit_nugget = 15000
                             )
        else:
            V =  skg.DirectionalVariogram(loocv_coords, loocv_values, 
                                          model = model, 
                                          azimuth = azimuth,
                                          tolerance=tolerance)
        
        # set up kriging instance
        ok = OrdinaryKriging(V, min_points=2, max_points=10, mode='exact')
        # interpolate left out point based on kriging
        loocv_rainfall_point = ok.transform(row["x_val"].flatten(), row["y_val"].flatten())
        # append the interpolated rainfall in one array
        rainfall_krig = np.append(rainfall_krig, loocv_rainfall_point)
        # calculate difference between interpolated and true rainfall for the respective point
        diff =  loocv_rainfall_point - row["rainfall"]
        # append residuals in one array
        residuals = np.append(residuals, diff)
    return rainfall_krig, residuals


def bubbleplot(x, y, residuals, ax=None, main = "Custom title"):
    """
    Function to plot kriging model residuals 
    based on the `seaborn` library which provides the `scatterplot()` function
    
    x,y
	arrays of coordinates
    residuals
	array of model residuals
    main
	title of the plot, default = "Custom title"
    ax 
	set axis to plot on, default = None
 
    """
 	
    import seaborn as sns
    ## data prep for nice plotting
    data = pd.DataFrame({"x": x, 
                    "y": y, 
                    "residuals": residuals})
    data['LOOCV_residuals'] = np.where(data['residuals'] > 0, 'neg', 'pos')
    data_pos = data[(data["residuals"] > 0)]
    data_neg = data[(data["residuals"] < 0)]
    
    if ax is None:
        ax = plt.gca()
    ax = sns.scatterplot(x=data["x"], y=data["y"], 
                size = abs(data["residuals"]), 
                hue =data["LOOCV_residuals"], 
                sizes=(1,300),
                legend=True, ax = ax, alpha = 0.7)
    plt.title(main)
    plt.xlabel("")
    plt.ylabel("")
    ## Legend outside of the plot
    plt.legend(title =" ",bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()

    return(ax)

