from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        result.extend(lst[i:i + n][::-1])
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}
    for s in lst:
        key = len(s)
        if key not in result:
            result[key] = []
        result[key].append(s)
    return result

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def recurse(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(recurse(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return recurse(nested_dict)

import itertools
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    return list(map(list, set(itertools.permutations(nums))))

import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates


from geopy.distance import geodesic

def decode_polyline(polyline_str):
    # Dummy function to simulate polyline decoding
    # Replace this with an actual polyline decoding library or logic
    return [(lat, lon) for lat, lon in zip(range(10), range(10))] 
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    points = decode_polyline(polyline_str)
    
    # Create lists for latitude and longitude
    latitudes = [point[0] for point in points]
    longitudes = [point[1] for point in points]
    
    # Calculate the distance between consecutive points
    distances = [0]  # No distance for the first point
    for i in range(1, len(points)):
        distances.append(geodesic(points[i-1], points[i]).meters)
    
    # Create a DataFrame with latitude, longitude, and distance columns
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Transpose the matrix and reverse each row to rotate 90 degrees clockwise
    rotated_matrix = [list(row) for row in zip(*matrix[::-1])]
    
    # Multiply each element by the sum of its original row and column index
    result_matrix = []
    for i, row in enumerate(matrix):
        result_row = []
        for j, val in enumerate(row):
            result_row.append(rotated_matrix[i][j] * (i + j))
        result_matrix.append(result_row)
    
    return result_matrix

dataset_1_path = 'D:\MapUp- Assessment 2\MapUp-DA-Assessment-2024\datasets\dataset-1.csv'
dataset_1 = pd.read_csv(dataset_1_path)
def time_check(dataset_1) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    dataset_1['timestamp'] = pd.to_datetime(dataset_1['timestamp'])
    
    # Check if each unique (`id`, `id_2`) has a full 7-day period and 24-hour coverage
    result = dataset_1.groupby(['id', 'id_2']).apply(lambda group: group['timestamp'].dt.date.nunique() == 7 and group['timestamp'].dt.hour.nunique() == 24)
    
    return result
