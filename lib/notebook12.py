""" """
from typing import Set, TypeVar, List, Dict, Tuple, Sequence, Any, ClassVar, TypeAlias
from dataclasses import dataclass, field
import pathlib
import re
from datetime import datetime
import numpy as np
import pandas as pd
import json
import bson
from ttp import ttp
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt # for Data visualization


T = TypeVar("T")
GenericCollection: TypeAlias = Set[T] | Tuple[T] | List[T]
NumType= TypeVar('NumType', int, float)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a pandas DataFrame by adjusting the data types of columns.

    This function optimizes the data types of numerical columns by downcasting them to the smallest
    possible types that still hold the column's values. This can help to fit larger DataFrames 
    into memory without requiring external libraries such as Dask, Ray, Modin or Vaex, or chunking.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose memory usage is to be optimized.
    verbose : bool, optional
        If True, prints the memory usage of the DataFrame before and after optimization, 
        as well as the percentage reduction in memory usage. Default is True.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with optimized data types.

    Notes
    -----
    - This function iterates through each column and adjusts the data type to a smaller one based on
      the minimum and maximum values in the column.
    - Only numerical columns are downcasted. Object (e.g., string) columns are left unchanged.
    - For integer columns, the function selects the smallest possible integer type (e.g., int8, int16)
      based on the value range of the column.
    - For floating-point columns, the function chooses the smallest float type that accommodates the 
      column's range of values (float16, float32, or float64).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [4.5, 5.5, 6.5],
    ...     'c': [100000, 200000, 300000]
    ... })
    >>> df_optimized = reduce_mem_usage(df)
    Memory usage of dataframe is 0.00 MB
    Memory usage after optimization is: 0.00 MB
    Decreased by 0.0%

    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
    if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def read_id_ifc_map(file_name: pathlib.Path):
    """
    Read a dictionary that maps optimization dam IDs to IFC dam IDs.

    Parameters
    ----------
    file_name : pathlib.Path
        The path to the JSON file containing the ID mapping.

    Returns
    -------
    dict
        A dictionary where the keys are optimization dam IDs (as integers) and 
        the values are corresponding IFC dam IDs.

    Notes
    -----
    This function reads a JSON file and converts the keys to integers, as JSON keys 
    are strings by default.

    Examples
    --------
    >>> id_ifc_map = read_id_ifc_map(pathlib.Path('path/to/id_to_ifc.json'))
    >>> print(id_ifc_map[1])  # Retrieve IFC ID for optimization dam ID 1
    """
    with open(file_name, 'r') as file:
        _id_ifc_map = json.load(file)
        return {int(key) : value for key, value in _id_ifc_map.items()}


def set_remap(
        value_list: Set[NumType] | str, value_map, missing_val_id: int = -99, 
        safe: bool = False) -> Set[NumType]:
    """
    Map values in a set to new values based on a provided mapping dictionary.

    Parameters
    ----------
    value_list : Set[NumType]
        A set of values to be mapped.
    value_map : dict
        A dictionary where the keys are original values, and the values are the 
        mapped values.
    missing_val_id : int, optional
        The value to assign when a value in `value_list` is not found in `value_map`. 
        Used only if `safe` is True. Default is -99.
    safe : bool, optional
        If True, values not found in `value_map` are replaced with `missing_val_id`. 
        If False, such values raise a KeyError. Default is False.

    Returns
    -------
    Set[NumType]
        A set of mapped values.

    Examples
    --------
    >>> value_list = {1, 2, 3}
    >>> value_map = {1: 'A', 2: 'B'}
    >>> set_remap(value_list, value_map, missing_val_id=-99, safe=True)
    {'A', 'B', -99}
    """
    if isinstance(value_list, str):
        values = list(map(int, value_list.strip('{}').split(', ')))
    else:
        values = value_list
    if safe:
        id_set = set([value_map.get(value, missing_val_id) for value in values])
        return id_set
    return {value_map[value] for value in values}


def get_every_n_row(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select every n-th row from a DataFrame.

    This function is useful for reducing the size of large DataFrames, containing
    continous portions of data, e.g. tightly-spaced time-series - aiding in data visualization 
    and prototype analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which rows will be selected.
    n : int
        The interval for row selection. Only every n-th row will be included in the result.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only every n-th row of the original DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
    >>> get_every_n_row(df, 2)
       A   B
    0  0  10
    2  2  12
    4  4  14
    6  6  16
    8  8  18
    """
    return df.iloc[::n]
    
    
@dataclass
class SolutionFileParser:
    """
    A class for parsing optimization solution `.sol` files generated by the 
    `moo_solver_CPAIOR` optimization algorithm.

    This class uses TTP (Template Text Processor), a Python module that allows 
    efficient parsing of semi-structured text data using templates.

    Attributes
    ----------
    file_path : str or pathlib.Path
        Path to the `.sol` file to be parsed.
    header_template : ClassVar[str]
        Template for parsing the header section of the `.sol` file.
    solution_template : str
        Template for parsing the solutions section of the `.sol` file. This 
        template is generated dynamically by the `_create_solution_template` method.
    header : str
        Raw text of the header section from the `.sol` file.
    solutions : str
        Raw text of the solutions section from the `.sol` file.
    data : dict
        Dictionary that holds parsed data, including the header and solutions 
        sections.

    Methods
    -------
    __post_init__()
        Reads the `.sol` file and extracts the header and solutions sections.
    parse()
        Parses the header and solutions sections using the provided templates 
        and stores the results in `self.data`.
    solutions_df()
        Converts the solutions data into a pandas DataFrame and computes additional 
        variables for analysis.
    to_json(json_file)
        Saves the parsed data to a JSON file.
    to_bson(file_path)
        Saves the parsed data to a BSON file.
    to_csv(csv_file)
        Saves the solutions data to a CSV file.
    """
    file_path: str | pathlib.Path
    # Template of the header text
    header_template: ClassVar[str] = """
Date/time: {{ date_time }}
Data file: {{ file_name }}
Wall time: {{ wall_time | to_float}} seconds.
CPU time: {{ cpu_time | to_float}} seconds.
seed: {{ seed | to_int}}
num_solutions: {{ num_solutions | to_int}}
# pruning steps (# nodes): {{ num_pruning_steps | to_int}}
Max policies considered: {{ max_policies | to_int}}
Policies considered: {{ num_policies | to_int}}
Pruned policies: {{ pruned_policies | to_int}}
epsilon: {{ epsilon | to_float}}
batch size: {{ batch_size | to_int }}
criteria: {{ criteria | ORPHRASE | split(',')}}
"""
    # Template for the solutions part of the .sol file 
    # (generated dynamically by `_create_solution_template` method)
    solution_template: str = ""
    header: str = ""
    solutions: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Initializes the `SolutionFileParser` by reading the specified `.sol` file and 
        extracting the header and solutions sections.

        Reads the raw file and stores the first 13 lines as `self.header` and 
        the remaining lines as `self.solutions`.
        """
        with open(self.file_path, 'r') as file:
            raw_data = file.readlines()
            self.header = "".join(raw_data[0:13])
            self.solutions = "".join(raw_data[14:])
            
    def _create_solution_template(self) -> None:
        """
        Dynamically creates a solution template for the TTP package.

        The template is based on the criteria specified in the header section, which 
        is parsed from the `.sol` file.
        """
        sol_template = ""
        crit_template = "{{{{ {} | to_float }}}}"
        num_dams_template = "{{{{ {} | to_int }}}}"
        for criterion in self.data['header']['criteria']:
            sol_template += crit_template.format(criterion) + ", "
        sol_template += num_dams_template.format('num_dams') + ", "
        sol_template += "{{ dam_ids | ORPHRASE | split(' ')}}"
        self.solution_template = sol_template
        
    def parse(self) -> None:
        """
        Parses the `.sol` file contents and stores the results in `self.data`.

        This method:
        1. Parses the header section using `self.header_template`.
        2. Dynamically creates a solution template based on the criteria in the header.
        3. Parses the solutions section using the dynamically generated solution template.
        
        Notes
        -----
        The `self.data` dictionary is populated with parsed data, including the 
        parsed header and solutions.
        """
        # 1. Parse header
        parser_header = ttp(self.header, self.header_template)
        parser_header.parse()
        header_data = parser_header.result(structure="flat_list")[0]
        header_data['criteria'] = [criterion.strip() for criterion in header_data['criteria']]
        date_time_formatted = re.sub(r'_+', ',', header_data['date_time'])
        header_data['date_time'] = datetime.strptime(
            date_time_formatted, "%a,%b,%d,%H,%M,%S,%Y")
        self.data['header'] = header_data
        # 2. Dynamically create a solution template
        self._create_solution_template()
        # 3. Parse solutions
        parser_sol = ttp(self.solutions, self.solution_template)
        parser_sol.parse()
        solution_data = parser_sol.result(structure="flat_list")
        for solution in solution_data:
            solution['dam_ids'] = [int(dam_id) for dam_id in solution['dam_ids']]
        self.data['solutions'] = solution_data
        
    @property
    def solutions_df(self) -> pd.DataFrame:
        """
        Converts the parsed solutions data into a pandas DataFrame and computes 
        additional variables for analysis.

        Additional variables include:
        - `land_loss`: Total land loss calculated as the sum of forest and 
          agricultural land loss.
        - `ghg_intensity`: Greenhouse gas emissions per energy produced, converted 
          to grams CO2 equivalent per kWh.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing solutions data with additional columns for 
            `land_loss` and `ghg_intensity`.
        """
        df = pd.DataFrame(self.data['solutions'])
        if set(['loss_agri', 'loss_forest']).issubset(set(df.columns)):
            df['land_loss'] = df['loss_agri'] + df['loss_forest']
        # Calculate ghg intensity, for ghg in tonneCO2eq/year and energy in MWh/d
        # GHG intensity needs to be in gCO2eq/kWh
        df['ghg_intensity'] = df['ghg'] / df['energy'] * 1_000 / 365.25 / 24
        return reduce_mem_usage(df)
        
        
    def to_json(self, json_file: str | pathlib.Path) -> None:
        """
        Saves the parsed data to a JSON file.

        Parameters
        ----------
        json_file : str or pathlib.Path
            Path to the JSON file where data will be saved.

        Notes
        -----
        The JSON file includes a custom serialization function to handle 
        datetime objects.
        """ 
        def serialize_datetime(obj: Any) -> str:
            """Custom serialization function for datetime objects"""
            if isinstance(obj, datetime):
                return obj.isoformat()
        # Save to json using custom datetime object serialization
        with open(json_file, 'w') as file:
            json_string = json.dumps(self.data, default=serialize_datetime, indent=4)
            file.write(json_string)
            
    def to_bson(self, file_path: str | pathlib.Path) -> None:
        """
        Saves the parsed data to a BSON file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the BSON file where data will be saved.
        """
        with open(file_path, 'wb') as bson_file:
            serialized_data = bson.dumps(self.data)
            bson_file.write(serialized_data)
            
    def to_csv(self, csv_file: str | pathlib.Path) -> None:
        """
        Saves the solutions data to a CSV file.

        Parameters
        ----------
        csv_file : str or pathlib.Path
            Path to the CSV file where solutions data will be saved.

        Notes
        -----
        The `dam_ids` column in the CSV file is saved as a string representation 
        of a list of integers. To retrieve the list format, parse this column with:
        `df.dam_ids = df.dam_ids.map(ast.literal_eval)`.
        """
        self.solutions_df.to_csv(csv_file, encoding='utf-8', index=False)
        
        
@dataclass
class OutputVisualiser:
    """
    Visualisation of optimization results from `moo_solver_CPAIOR`.

    Attributes
    ----------
    data : pd.DataFrame
        Optimization outputs.
    """
    data: pd.DataFrame
        
    @property
    def columns(self) -> List[str]:
        """
        Return the list of column names.

        Returns
        -------
        list of str
            List of column names in the DataFrame.
        """
        return list(self.data.columns)
    
    def plot_parallel(
            self, columns: GenericCollection[str], color_col: str, title: str | None = None,
            color_limits: Tuple[float, float] = (0,200), **kwargs) -> go.Figure:
        """
        Plot a parallel coordinate plot using plotly.

        Parameters
        ----------
        columns : GenericCollection[str]
            List of columns (variables) to be plotted.
        color_col : str
            Column which values should be associated with line color.
        title : str, optional
            Plot title, by default None.
        color_limits : Tuple[float, float], optional
            Tuple with minimum and maximum value associated with the minimum and maximum
            color value in the color palette, by default (0, 200).
        **kwargs : 
            Additional keyword arguments for plotly express `parallel_coordinates`.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly figure object for the parallel coordinate plot.
        """
        # Other scales:
        # px.colors.diverging.Tealrose
        # px.colors.sequential.Blues
        # px.colors.sequential.Oranges
        # px.colors.diverging.RdYlBu
        # color_continuous_scale=px.colors.diverging.Armyrose
        # color_continuous_midpoint=2
        fig = px.parallel_coordinates(self.data, color=color_col, dimensions=columns,
                              color_continuous_scale=px.colors.diverging.Tealrose, width=1000,
                              title=title, range_color=color_limits, **kwargs)
        fig.show()
        return fig
        
    def plot_scatter_2D(
            self, x_col: str, y_col: str, hue: str | None = None, 
            size: str | None = None, palette: str = "hot", 
            xlabel: str | None = None, ylabel: str | None = None) -> go.Figure:
        """
        Plot a 2D scatter plot with optimization outputs.

        Parameters
        ----------
        x_col : str
            Data (column) to be placed on the x-axis.
        y_col : str
            Data (column) to be placed on the y-axis.
        hue : str, optional
            Data (column) associated with hue, by default None.
        size : str, optional
            Data (column) associated with marker size, by default None.
        palette : str, optional
            Plotly.express color palette object, by default "hot".
        xlabel : str, optional
            X-axis label text, by default None.
        ylabel : str, optional
            Y-axis label text, by default None.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly figure object for the 2D scatter plot.
        """
        kwargs  =   {
             'edgecolor' : "k",
             'facecolor' : "w",
             'linewidth' : 0.2,
             'linestyle' : '-',
            }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_style('white')
        sns.set_context("paper", font_scale = 1)
        sns.despine(right = True)
        sns.scatterplot(
            x = x_col, y = y_col, data = self.data, hue=hue, palette=palette, size=size,
            marker = 'o', **kwargs, alpha = 0.95)
        ax.legend(title='Development scenario / Firm Energy, MW', fontsize=10, 
                  title_fontsize=12, frameon=False,
                  ncol=3)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        fig.show()
        return fig


@dataclass
class ObjectiveCalculator:
    """
    A class to calculate the sum of objective values for selected dams based on their 
    associated IFC IDs from an objective dataframe.

    Attributes
    ----------
    obj_df : pd.DataFrame
        A pandas DataFrame containing the objective values for the dams.
    ids : list of NumType, optional
        A list of dam IDs (IFC_IDs) that are used in processing objectives. Defaults to an empty list.
    obj_names : list of str
        A class-level list of objective value column names, including:
        'HP_mean', 'HP_firm', 'tot_em', 'crop_area_loss_km2', 'forest_area_loss_km2'.
    """
    obj_df: pd.DataFrame
    ids: List[NumType] = field(default_factory = list)
    obj_names: ClassVar[List[str]] = [
        'HP_mean', 'HP_firm', 'tot_em', 'crop_area_loss_km2', 
        'forest_area_loss_km2']
    
    def _filter_df(self) -> Tuple[pd.DataFrame, List[NumType]]:
        """
        Filter the objective dataframe to select rows corresponding to the provided 
        dam IDs and return the filtered dataframe along with the list of missing IDs.

        If no missing indices are found, the returned list will be empty.

        Returns
        -------
        Tuple
            - pd.DataFrame : A dataframe filtered to include only the rows 
              corresponding to the provided dam IDs.
            - list of NumType : A list of IDs that were not found in the dataframe.
        """
        filtered_df = self.obj_df[self.obj_df.index.isin(self.ids)]
        found_indices = filtered_df.index.to_list()
        missed_indices = set(self.ids) - set(found_indices)
        return filtered_df, list(missed_indices)
    
    @property
    def objectives(self) -> pd.Series:
        """
        Calculate and return the sum of objective values for the selected dams 
        based on the IFC IDs provided in the `ids` attribute.

        Returns
        -------
        pd.Series
            A pandas Series containing the sum of the objective values for the 
            selected dams, indexed by the objective names.
        """
        return self._filter_df()[0][self.obj_names].sum()

    
def map_ids(moo_ids: Sequence[int], id_map: Dict[int, int]) -> Sequence[int]:
    """
    Map a sequence of IDs to new values using a dictionary.

    This function takes a sequence of IDs (e.g., dam IDs) and returns a new sequence 
    where each original ID is replaced by its corresponding value in the `id_map` dictionary.
    If an ID is not found in the dictionary, it remains unchanged in the output.

    Parameters
    ----------
    moo_ids : Sequence[int]
        A sequence of IDs (e.g., dam IDs) to be mapped.
    id_map : Dict[int, int]
        A dictionary that maps the original IDs to the new IDs.

    Returns
    -------
    Sequence[int]
        A sequence of mapped IDs, where each original ID is replaced by its corresponding 
        value in the `id_map`, or remains unchanged if not found in the map.
    """
    return [id_map.get(item, item) for item in moo_ids]


def find_solution_by_dam_numbers(df: pd.DataFrame, dam_ids: Set[int]) -> pd.DataFrame:
    """
    Find rows in a dataframe corresponding to the selected dams by their IDs.

    This function filters the given dataframe to return rows where the 'dam_ids' column
    matches any of the IDs in the `dam_ids` set.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe containing optimization results, where each row corresponds to a dam.
    dam_ids : Set[int]
        A set of dam IDs that need to be matched in the dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the rows where the 'dam_ids' column matches any of the specified
        dam IDs.
    """
    return df[df['dam_ids']] == dam_ids

def return_row_by_criterion(df: pd.DataFrame, criterion: str, value: Any) -> pd.Series:
    """
    Find the row with the closest value to the specified criterion value.

    This function finds the row in the dataframe where the value in the `criterion` column 
    is closest to the specified `value`. The absolute difference between the `criterion` 
    column values and the provided `value` is calculated, and the row with the smallest 
    difference is returned.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe in which the rows contain data related to the selected criterion.
    criterion : str
        The name of the column (criterion) for which to find the closest value.
    value : Any
        The target value to which the `criterion` values are compared.

    Returns
    -------
    pd.Series
        A pandas Series representing the row with the closest value in the specified `criterion` column.
    """
    # Calculate the absolute differences
    differences = (df[criterion] - value).abs()
    # Find the index of the minimum absolute difference
    closest_index = differences.idxmin()
    # Return the row using the index
    closest_row = df.loc[closest_index]
    return closest_row


if __name__ == "__main__":
    pass
