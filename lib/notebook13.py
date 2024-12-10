"""Helper functions supporting Notebook_13"""
import matplotlib
from math import pi
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point
import matplotlib.gridspec as gridspec


def calc_composite_metrics_radar_df(df: pd.DataFrame):
    sums = df[['Loss of forest, [%]',
       'Loss of agricultural land, [%]', 'GHG emissions, [%]',
       'HP Production, [%]']].sum()
    averages = df[['Firm Power Ratio, [%]']].mean()
    summary_series = pd.concat([sums, averages], axis=0)
    summary_series.index = [
        'Forest\nloss', 'Agri\nland\nloss', 'GHG\nEmissions', 'HP', 'Firm\nPower']
    return summary_series


class SeabornFig2Grid:
    """
    Embed a Seaborn grid (FacetGrid, PairGrid, or JointGrid) into a specified 
    subplot area within a Matplotlib figure.

    This class allows for integrating complex Seaborn grids directly into 
    a larger figure with subplots, making it easier to compose multiple 
    visualizations into a single, unified layout.

    Parameters
    ----------
    seaborngrid : sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, or sns.axisgrid.JointGrid
        The Seaborn grid to embed within the Matplotlib figure.
    fig : matplotlib.figure.Figure
        The target Matplotlib figure where the Seaborn grid will be embedded.
    subplot_spec : matplotlib.gridspec.SubplotSpec
        The gridspec subplot specification indicating where in the figure the 
        Seaborn grid should be placed.
    """
    def __init__(self, seaborngrid, fig,  subplot_spec) -> None:
        """
        Initialize the SeabornFig2Grid instance and embed the specified Seaborn grid.

        Parameters
        ----------
        seaborngrid : sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, or sns.axisgrid.JointGrid
            The Seaborn grid to embed within the Matplotlib figure.
        fig : matplotlib.figure.Figure
            The Matplotlib figure to embed the Seaborn grid into.
        subplot_spec : matplotlib.gridspec.SubplotSpec
            The gridspec subplot specification indicating where to place the grid.
        """
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self) -> None:
        """
        Embed a PairGrid or FacetGrid into the specified subplot area of the Matplotlib figure.
        
        This method creates a new gridspec for the Seaborn grid and moves each subplot
        to the appropriate position within the target figure.
        """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self) -> None:
        """
        Embed a JointGrid into the specified subplot area of the Matplotlib figure.
        
        This method adjusts the gridspec to accommodate both the main plot and
        marginal plots of a Seaborn JointGrid.
        """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs) -> None:
        """
        Move an axis to a specified gridspec location within the Matplotlib figure.

        This function reassigns an axis to the figure and updates its position 
        and subplot specification.
        
        https://stackoverflow.com/a/46906599/4124317

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to move.
        gs : matplotlib.gridspec.SubplotSpec
            The target gridspec location to move the axis into.
        """
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self) -> None:
        """
        Finalize the embedding by closing the original Seaborn figure and
        setting up resize events to maintain the layout in the new figure.
        
        Connects a resize event to the Matplotlib canvas to adjust the embedded
        Seaborn grid when the figure size changes.
        """
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None) -> None:
        """
        Resize the embedded Seaborn grid to match the current size of the Matplotlib figure.

        Parameters
        ----------
        evt : matplotlib.backend_bases.Event, optional
            The resize event that triggered the function call (default is None).
        """
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        

def plot_mya_reservoirs(
        data: pd.DataFrame,
        ax: matplotlib.axes.Axes,
        column_name: str,
        lon_field: str = "coordinates_1",
        lat_field: str = "coordinates_0",
        rivers_shp: str | None = r'bin/gis_layers/mya_rivers.shp',
        outline_shp: str | None = r'bin/gis_layers/myanmar_outline/Myanmar_outline.shp',
        res_label_field: str | None = None,
        title: str | None = None,
        title_font_size: int = 14,
        marker_size: int | str = 40,
        marker_size_multiplier: float = 1.0,
        cmap = 'tab10',
        plot_legends = (False, False, False),
        offset: float = 0.0,
        legend_y_coords = (1.0, 0.2, -0.05),
        **kwargs):
    """Plots the maps with dams"""
    crs={'init':'epsg:4326'}
    geometry=[Point(xy) for xy in zip(data[lon_field], data[lat_field])]
    data_gdf=gpd.GeoDataFrame(data,crs=crs, geometry=geometry)
    if rivers_shp is not None:
        rivers = gpd.read_file(rivers_shp, crs=crs)
        rivers.plot(ax=ax, edgecolor = 'k', linewidth=0.3, alpha = 0.5)
    if outline_shp is not None:
        outline = gpd.read_file(outline_shp)
        outline.plot(ax=ax, facecolor='Grey', edgecolor='k',alpha=0.05,linewidth=1,cmap="cividis")
        outline.plot(ax=ax, facecolor='none', edgecolor='k',alpha=1,linewidth=0.2,cmap="cividis")
    # Divide dams into storage and RoR
    data_gdf_ror = data_gdf[data_gdf['hp_type_reem'] == 'ror']
    data_gdf_sto = data_gdf[data_gdf['hp_type_reem'] == 'sto']
    
    data_gdf_sto_hp = data_gdf[(data_gdf['hp_type_reem'] == 'sto') & (data_gdf['type_y'] == 'hydroelectric')]
    data_gdf_sto_mp = data_gdf[(data_gdf['hp_type_reem'] == 'sto') & (data_gdf['type_y'] == 'multipurpose')]
    
    if isinstance(marker_size, str):
        marker_size_ror = np.sqrt(data_gdf_ror[marker_size].astype(float))
        marker_size_sto = np.sqrt(data_gdf_sto[marker_size].astype(float))
        marker_size_sto_hp = np.sqrt(data_gdf_sto_hp[marker_size].astype(float))
        marker_size_sto_mp = np.sqrt(data_gdf_sto_mp[marker_size].astype(float))
        
    p1 = data_gdf_ror.plot(
        kind='geo', ax=ax, column=column_name, cmap=cmap, 
        markersize=marker_size_ror*marker_size_multiplier,
        marker = "^", k=10,
        linewidth=0.5, edgecolor='k', alpha=0.45, categorical = True, 
        legend_kwds={
            'loc':'upper right', 
            'bbox_to_anchor': (1.17+offset, legend_y_coords[0]),
            'markerscale':0.5, 
            'title_fontsize':'medium', 
            'frameon':False,
            'fontsize':"small"},
        legend=plot_legends[0],
        **kwargs)
    p2 = data_gdf_sto_hp.plot(
        kind = 'geo', ax=ax, column=column_name, cmap=cmap, 
        markersize=marker_size_sto_hp*marker_size_multiplier, 
        marker = "o", k = 10,
        linewidth=0.5, edgecolor='k', alpha=0.45, categorical = True,        
        **kwargs)
    p3 = data_gdf_sto_mp.plot(
        kind = 'geo', ax=ax, column=column_name, cmap=cmap, aspect=1,
        markersize=marker_size_sto_mp*marker_size_multiplier, 
        marker = "s", k = 10,
        linewidth=0.5, edgecolor='k', alpha=0.45, categorical = True, **kwargs)
    
    if plot_legends[0]:
        leg1 = p1.get_legend()
        leg1._legend_box.align = "left"
        custom_labels = ["0.0-3.1", "3.1-20", "20-40", "40-60", "60-100", "100-400", "400-800", "800-5000"]
        if leg1:
            leg1.set_title("GHG Emission\nIntensity\n[gCO$_{2,eq}$/kWh]")
        if leg1 and custom_labels:
            new_legtxt = custom_labels
            for ix,eb in enumerate(leg1.get_texts()):
                eb.set_text(new_legtxt[ix])
        ax.add_artist(leg1)
    
    if plot_legends[1]:
        legend_elements = [
            Line2D([0], [0], marker='o', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color='none',
                   label='Sto HP', markerfacecolor='none', markersize=10),
            Line2D([0], [0], marker='s', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color="none",
                   label='Sto Multipurpose', markerfacecolor='none', markersize=10),
            Line2D([0], [0], marker='^', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color='none',
                   label='RoR', markerfacecolor='none', markersize=10)]
        ax.legend(handles=legend_elements, **{
                'loc':'upper right', 
                'bbox_to_anchor': (1.2+offset, legend_y_coords[1]),
                'markerscale':1, 
                'title_fontsize':'medium', 
                'frameon':False,
                'fontsize':"small"})
        leg2 = ax.get_legend()
        leg2._legend_box.align = "left"
        leg2.set_title("HP Type")
        ax.add_artist(leg2)
    if plot_legends[2]:
        legend_elements = [
            Line2D([0], [0], marker='o', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color='none',
                   label='10MW', markerfacecolor='none', markersize=4),
            Line2D([0], [0], marker='o', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color="none",
                   label='100MW', markerfacecolor='none', markersize=10),
            Line2D([0], [0], marker='o', markeredgecolor='k', markeredgewidth=0.2, alpha=1,
                   color='none',
                   label='1000MW', markerfacecolor='none', markersize=20)]
        ax.legend(handles=legend_elements, **{
                'loc':'upper right', 
                'bbox_to_anchor': (1.2+offset, legend_y_coords[2]),
                'markerscale':1.00, 
                'title_fontsize':'medium', 
                'frameon':False,
                'fontsize':'small'})
        leg3 = ax.get_legend()
        leg3._legend_box.align = "left"
        leg3.set_title("Mean Power Output")
    
    #You can use different 'cmaps' such as jet, plasm,magma, infereno,cividis, binary...(I simply chose cividis)
    ax.set_yticks([])
    ax.set_xticks([])
    x_spacing = 3
    y_spacing = 3
    #ax.set_xlim(data_gdf.total_bounds[0] - x_spacing/2, data_gdf.total_bounds[2] + x_spacing/2)
    #ax.set_ylim(data_gdf.total_bounds[1] - y_spacing/2, data_gdf.total_bounds[3] + y_spacing/2)
    if title is not None:
        ax.set_title(title, fontsize=title_font_size, loc='left', y=1.0, pad=-18)
    # Remove axes
    for pos in ('top', 'right', 'bottom', 'left'):
        ax.spines[pos].set_visible(False)
    # Label the reservoirs
    if res_label_field is not None:
        for x, y, label in zip(data['coordinates_1'], data['coordinates_0'], data[res_label_field]):
            ax.annotate(label, xy=(x,y), xytext=(4,4), textcoords='offset points', fontsize=5, alpha=0.7)

    return p1, p2, p3


def make_radar_plots(axs, df_radar: pd.DataFrame, optim_scenarios: pd.DataFrame) -> None:
    """ """
    optim_scenarios_keys = list(optim_scenarios.keys())
    for ix, (scenario_id, dams) in enumerate(optim_scenarios.items()):
        """ """
        if ix * 2 + 1 > len(optim_scenarios_keys):
            break
        key1 = optim_scenarios_keys[ix * 2]
        key2 = optim_scenarios_keys[ix * 2 + 1]
        dams1 = optim_scenarios[key1]
        dams2 = optim_scenarios[key2]

        marker_size = 10
        # Calculate the number of categories in the radar plot
        
        
        categories=list(df_radar)[:]
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1] 
        
        scenarios = ("I", "II", "III")
        
        ax = axs[ix]

        handles = [
            Line2D([], [],  lw=0.4, c=color, marker="o", markersize=marker_size/6, label=species)
            for species, color in zip(["Built", "Not Built"], ['tab:blue', 'tab:orange'])
        ]

        if ix == 2:
            legend = ax.legend(
                handles=handles,
                prop={'size': 8},
                loc=(1, 0),       # bottom-right
                labelspacing=0.5, # add space between labels
                frameon=False     # don't put a frame
            )
            
        data1 = df_radar.loc[dams1]
        data1_summary = calc_composite_metrics_radar_df(data1)
        data2 = df_radar.loc[dams2]
        data2_summary = calc_composite_metrics_radar_df(data2)
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        
        categories_abbr = list(data1_summary.index)
        ax.set_xticks(angles[:-1], categories_abbr, size=8)

        # First series
        values1=data1_summary.values.flatten().tolist()
        values1 += values1[:1]
        # Plot the first series
        ax.plot(angles, values1, linewidth=0.75, linestyle='solid', label="Built", alpha=0.5)
        ax.fill(angles, values1, 'tab:blue', alpha=0.1)
        ax.scatter(angles, values1, s=marker_size, zorder=10)
        # Second series
        values2=data2_summary.values.flatten().tolist()
        values2 += values2[:1]
        ax.plot(angles, values2, linewidth=0.75, linestyle='solid', label="Not Built", alpha=0.5)
        ax.fill(angles, values2, 'tab:orange', alpha=0.1)
        ax.scatter(angles, values2, s=marker_size, zorder=10)

        max_value = max(values1 + values2)
        #upper_limit = min([value for value in ytick_values if value > max_value])
        upper_limit = max_value
        ax.set_ylim(0, upper_limit)

        ax.set_xticks(angles[:-1])
        ax.tick_params(pad=3, size=7)
        
        ax.spines["start"].set_color("none")
        
        HANGLES = np.linspace(0, 2 * np.pi)
        
        H0 = np.zeros(len(HANGLES))
        H1 = np.ones(len(HANGLES)) * 20
        H2 = np.ones(len(HANGLES)) * 40
        H3 = np.ones(len(HANGLES)) * 60
        H4 = np.ones(len(HANGLES)) * 80
        H5 = np.ones(len(HANGLES)) * 100
        
        if upper_limit == 80:
            Hs = {0: H0, 40: H2, 80: H4}
        elif upper_limit == 100:
            Hs = {0: H0, 20: H1, 60: H3, 100:H5}
        else:
            Hs = {0: H0, 20: H1, 40: H2, 60: H3, 100:H5}

        for key, hangle in Hs.items():
            if key > upper_limit:
                break
            ax.plot(HANGLES, hangle, lw=0.8, ls=(0, (5, 6)), c='Grey')

        text_angle = -37.1
        PAD = 2
        
        ax.set_title(
            scenarios[ix], fontsize=14, loc='left', y=1.2, pad=-2, fontdict={'fontstyle': 'italic'})
        ax.tick_params(axis='x', labelsize=9)
        
        for level in Hs:
            if level == 0:
                continue
            if level > upper_limit:
                break
            text = f"{level}%"
            ax.text(text_angle, level + PAD, text, size=6, alpha=0.7)
    return


def plot_rivord_histograms(axs, output_df: pd.DataFrame, optim_scenarios: pd.DataFrame):
    """ """
    optim_scenarios_keys = list(optim_scenarios.keys())

    for ix, (scenario_id, dams) in enumerate(optim_scenarios.items()):
        """ """
        if ix * 2 + 1 > len(optim_scenarios_keys):
            break
        key1 = optim_scenarios_keys[ix * 2]
        key2 = optim_scenarios_keys[ix * 2 + 1]
        dams1 = optim_scenarios[key1]
        dams2 = optim_scenarios[key2]

        data1 = output_df.loc[dams1]['RIV_ORD'].to_numpy()
        data2 = output_df.loc[dams2]['RIV_ORD'].to_numpy()

        combined_data = np.concatenate([data1, data2])
        # Determine the bins based on the combined data
        bins = np.histogram_bin_edges(combined_data, bins=8)
        bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

        ax=axs[ix]
        hp_dist_colors = ["tab:blue", "tab:orange"]
        ax.hist([data1, data2], bins=bins, alpha = 0.5, edgecolor='black', linewidth=0.3,
                label=['Built', 'Not Built'], color=hp_dist_colors)
        
        if ix == 0:
            ax.set_ylim(0, 20)
        else:
            ax.set_ylim(0, 32)
        ax.set_xlim(2,8)
        ax.set_ylabel("Number of sites", fontsize=8)
        ax.set_xlabel("River Order", fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xticks([2,3,4,5,6,7,8])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ix * 2 + 1 == len(optim_scenarios_keys) - 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor = (1.0, 1.1), 
                      prop={'size': 8})


def plot_histograms(axs, output_df: pd.DataFrame, optim_scenarios: pd.DataFrame) -> None:
    """ """
    optim_scenarios_keys = list(optim_scenarios.keys())

    for ix, (scenario_id, dams) in enumerate(optim_scenarios.items()):
        """ """
        if ix * 2 + 1 > len(optim_scenarios_keys):
            break
        key1 = optim_scenarios_keys[ix * 2]
        key2 = optim_scenarios_keys[ix * 2 + 1]
        dams1 = optim_scenarios[key1]
        dams2 = optim_scenarios[key2]

        data1 = output_df.loc[dams1]['HP Production [GWh/year]'].to_numpy()
        data2 = output_df.loc[dams2]['HP Production [GWh/year]'].to_numpy()

        combined_data = np.concatenate([data1, data2])
        # Determine the bins based on the combined data
        bins = np.histogram_bin_edges(combined_data, bins=8)
        logbins=np.logspace(np.log10(1),np.log10(10000), 10)
        
        #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

        ax=axs[ix]
        hp_dist_colors = ["tab:blue", "tab:orange"]
        ax.hist([data1, data2], bins=logbins, alpha = 0.5, edgecolor='black', linewidth=0.3,
                label=['Built', 'Not Built'], color=hp_dist_colors)
        
        if ix == 0:
            ax.set_ylim(0, 15)
        else:
            ax.set_ylim(0, 20)
        ax.set_xlim(10, 10_000)
        #ax.set_xscale('log')
        ax.set_ylabel("Number of sites", fontsize=8)
        ax.set_xlabel("HP Production [GWh/year]", fontsize=8)
        ax.set_xticks([10,100,1_000,10_000])
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ix * 2 + 1 == len(optim_scenarios_keys) - 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor = (1.0, 1.1), 
                      prop={'size': 8})
                      

def plot_elev_dist(axs, output_df: pd.DataFrame, optim_scenarios: pd.DataFrame, df_combined: pd.DataFrame) -> None:
    """ """
    optim_scenarios_keys = list(optim_scenarios.keys())
    
    ix=2
    # Use the same bins for each plot
    key1 = optim_scenarios_keys[ix * 2]
    key2 = optim_scenarios_keys[ix * 2 + 1]
    dams1 = optim_scenarios[key1]
    dams2 = optim_scenarios[key2]

    data1 = df_combined.loc[dams1]['FSL (m)'].to_numpy()
    data2 = df_combined.loc[dams2]['FSL (m)'].to_numpy()

    combined_data = np.concatenate([data1, data2])
    # Determine the bins based on the combined data
    bins = np.histogram_bin_edges(combined_data, bins=10)    

    for ix, (scenario_id, dams) in enumerate(optim_scenarios.items()):
        """ """
        if ix * 2 + 1 > len(optim_scenarios_keys):
            break
        key1 = optim_scenarios_keys[ix * 2]
        key2 = optim_scenarios_keys[ix * 2 + 1]
        dams1 = optim_scenarios[key1]
        dams2 = optim_scenarios[key2]
        
        data1 = df_combined.loc[dams1]['FSL (m)'].to_numpy()
        data2 = df_combined.loc[dams2]['FSL (m)'].to_numpy()
        
        combined_data = np.concatenate([data1, data2])
        # Determine the bins based on the combined data
        #bins = np.histogram_bin_edges(combined_data, bins=10)

        ax=axs[ix]
        elev_dist_colors = ["tab:blue", "tab:orange"]
        ax.hist([data1, data2], bins=bins, alpha = 0.5, edgecolor='black', linewidth=0.3,
                label=['Built', 'Not Built'], color=elev_dist_colors)
        if ix == 0:
            ax.set_ylim(0, 15)
        else:
            ax.set_ylim(0, 25)
        ax.set_xlim(0, 1500)
        ax.set_ylabel("Number of sites", fontsize=8)
        ax.set_xlabel("Elevation, m.a.s.l.", fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ix * 2 + 1 == len(optim_scenarios_keys) - 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor = (1, 1.1),
                     prop={'size': 8})   

    
if __name__ == "__main__":
    """ """
