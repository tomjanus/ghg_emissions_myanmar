""" """
from typing import Tuple, List
import math
import plotly as px
import matplotlib
from shapely.geometry import Point
import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from bokeh.io import output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool, CategoricalColorMapper
from bokeh.palettes import brewer
from bokeh.palettes import Spectral6
from bokeh.palettes import RdBu3


def _simple_px_plot() -> None:
    """UNUSED"""
    fig1 = px.scatter_geo(
        data_full, lat='coordinates_0', lon='coordinates_1', color='cluster',
        color_discrete_sequence=px.colors.qualitative.G10,
        #color_continuous_scale='reds',
        #color_discrete_sequence=["red", "green", "blue", "magenta", "yellow"],
        size_max=15,
        hover_name='Reservoir', size='res_mean_depth',
        #symbol='type',
        title='Reservoir clusters')
    fig1.show()
    

def plot_mya_reservoirs_gdf(
        data: pd.DataFrame,
        lon_field: str = "coordinates_1",
        lat_field: str = "coordinates_0",
        cluster_field: str = 'cluster',
        tooltip_fields: Tuple[str, ...] = ('Reservoir', 'cluster', 'res_area', 'res_mean_depth')) -> folium.folium.Map:
    """ """
    crs={'init':'epsg:4326'}
    geometry=[Point(xy) for xy in zip(data[lon_field], data[lat_field])]
    data_gdf=gpd.GeoDataFrame(data,crs=crs, geometry=geometry)
    
    if 'probability' in data.columns:
        styling_fun = lambda x: {
            "radius": 2 * math.log(x["properties"]["res_mean_depth"]),
            "fillOpacity": x["properties"]['probability']}
    else:
        styling_fun = lambda x: {
            "radius": 2 * math.log(x["properties"]["res_mean_depth"])}        
    
    res_map = data_gdf.explore(
        column=cluster_field,
        legend=True,
        tooltip=tooltip_fields,
        style_kwds =dict(color="black", weight=0.7, radius = 5, fillOpacity=0.7, 
                        style_function = styling_fun),
        legend_kwds=dict(colorbar=True))
    return res_map


def plot_mya_reservoirs_static(
        data: pd.DataFrame,
        ax: matplotlib.axes._axes.Axes,
        lon_field: str = "coordinates_1",
        lat_field: str = "coordinates_0",
        rivers_shp: str | None = r'bin/gis_layers/mya_rivers.shp',
        outline_shp: str | None = r'bin/gis_layers/myanmar_outline/Myanmar_outline.shp',
        res_label_field: str | None = None,
        title: str | None = None,
        marker_size: int | str = 40,
        marker_size_multiplier: float = 1.0):
    """ """
    crs={'init':'epsg:4326'}
    geometry=[Point(xy) for xy in zip(data[lon_field], data[lat_field])]
    data_gdf=gpd.GeoDataFrame(data,crs=crs, geometry=geometry)
    if rivers_shp is not None:
        rivers = gpd.read_file(rivers_shp)
        rivers.plot(ax=ax, edgecolor = 'blue', linewidth=0.6, alpha = 0.5)
    if outline_shp is not None:
        outline = gpd.read_file(outline_shp)
        outline.plot(ax=ax, facecolor='Grey', edgecolor='k',alpha=0.1,linewidth=1,cmap="cividis")
    if isinstance(marker_size, str):
        marker_size = data_gdf[marker_size].astype(float)
    data_gdf.plot(
        ax=ax, column='cluster', cmap='coolwarm', markersize=marker_size*marker_size_multiplier, 
        linewidth=0.2, edgecolor='k', alpha=0.7, categorical = True, legend = True)
    #You can use different 'cmaps' such as jet, plasm,magma, infereno,cividis, binary...(I simply chose cividis)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Label the reservoirs
    if res_label_field is not None:
        for x, y, label in zip(data['geometry'].x, data['geometry'].y, data[res_label_field]):
            ax.annotate(label, xy=(x,y), xytext=(4,4), textcoords='offset points', fontsize=5, alpha=0.7)
            
            
def plot_with_bokeh(
        data: pd.DataFrame,
        lon_field: str = "coordinates_1",
        lat_field: str = "coordinates_0",
        marker_size: int | str = 20,
        marker_size_multiplier: float = 1.0,
        outline_shp: str | None = r'bin/gis_layers/myanmar_outline/Myanmar_outline.shp',
        fig_dim: Tuple[int, int] = (500, 900),
        tooltips: List[Tuple[str, str]] | None = None,
        title: str | None = None):
    """ """
    Tooltips = List[Tuple[str, str]]
    def add_at_symbols(tooltips: Tooltips) -> Tooltips:
        return [(first, '@' + second) for first, second in tooltips]

    output_notebook()
    crs={'init':'epsg:4326'}
    geometry=[Point(xy) for xy in zip(data[lon_field], data[lat_field])]
    data_gdf=gpd.GeoDataFrame(data,crs=crs, geometry=geometry)
    data_gdf['cluster_str'] = data_gdf['cluster'].astype(str)
    factors = list(sorted(data_gdf['cluster_str'].unique()))
    
    if isinstance(marker_size, str):
        data_gdf['radius'] = np.sqrt(data_gdf[marker_size]*marker_size_multiplier)
    else:
        data_gdf['radius'] = marker_size * marker_size_multiplier
        
    res_geojson=GeoJSONDataSource(geojson=data_gdf.to_json())
    
    if outline_shp is not None:
        outline_gdf = gpd.read_file(outline_shp)
        outline_geojson=GeoJSONDataSource(geojson=outline_gdf.to_json())
    
    color_mapper = CategoricalColorMapper(factors=factors, palette=RdBu3[:len(factors)])
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                         border_line_color=None,location = (0,0), orientation = 'horizontal')
    
    if title is None:
        p = figure(
            width=fig_dim[0], height=fig_dim[1], x_axis_type="linear", y_axis_type="linear")
    else:
        p = figure(
            width=fig_dim[0], height=fig_dim[1], title=title, x_axis_type="linear", y_axis_type="linear")        

    country = p.patches(source=outline_geojson, fill_color='grey', alpha=0.2)
    reservoirs = p.circle(x=lon_field, y=lat_field,
             size='radius',
             fill_color = {'field' :'cluster_str', 'transform' : color_mapper},
             line_color="black",
             fill_alpha=0.4,
             source=res_geojson)

    if tooltips is not None:
        # Process tooltips (add the 'at' symbol)
        tooltips = add_at_symbols(tooltips)
        p.add_tools(HoverTool(tooltips = tooltips, renderers=[reservoirs]))
    
    p.add_layout(color_bar, 'below')
    show(p)
