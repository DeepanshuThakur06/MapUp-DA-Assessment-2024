import pandas as pd
import numpy as np
dataset_1_path='D:\MapUp- Assessment 2\MapUp-DA-Assessment-2024\datasets\dataset-1.csv'
dataset_2_path='D:\MapUp- Assessment 2\MapUp-DA-Assessment-2024\datasets\dataset-2.csv'
dataset_1 = pd.read_csv(dataset_1_path)
dataset_2 = pd.read_csv(dataset_2_path)

def calculate_distance_matrix() -> pd.DataFrame:
    """
    Calculates a distance matrix based on the distance data from dataset_2,
    excluding null values and using zero for missing distances.
    """
    # Pivoting dataset_2 to create a distance matrix
    distance_matrix = dataset_2.pivot(index='id_start', columns='id_end', values='distance')
    
    # Fill NaN values with 0
    distance_matrix = distance_matrix.fillna(0)

    return distance_matrix

# Calculate and print the distance matrix
distance_matrix = calculate_distance_matrix()
print("Distance Matrix:")
print(distance_matrix)

def unroll_distance_matrix() -> pd.DataFrame:
#     """
#     Unroll a distance matrix to a DataFrame in the style of the initial dataset.

#     Args:
#         df (pandas.DataFrame)

#     Returns:
#         pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
#     """
#     # Write your logic here
    unrolled_df = dataset_2.copy()

    # Return the unrolled DataFrame directly, since dataset_2 already has the desired format
    return unrolled_df

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix()  # No argument needed now

print("Unrolled Distance DataFrame:")
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(reference_id, reference_distance) -> pd.Series:
    """
    Finds IDs from dataset_2 that have distances within a 10% threshold of the reference distance
    from a specified reference ID.

    Args:
        reference_id (int): The ID to compare against.
        reference_distance (float): The distance threshold to compare.

    Returns:
        pd.Series: A series containing IDs that meet the criteria.
    """
    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_distance * 0.90
    upper_bound = reference_distance * 1.10

    # Filter dataset_2 for relevant distances
    filtered_ids = dataset_2[
        (dataset_2['id_start'] == reference_id) & 
        (dataset_2['distance'] >= lower_bound) & 
        (dataset_2['distance'] <= upper_bound)
    ]['id_end']  # Assuming we want the corresponding end IDs

    return filtered_ids.reset_index(drop=True)

# Example usage
reference_id = 1001400  # Example reference ID
reference_distance = 5  # Example reference distance

ids_within_threshold = find_ids_within_ten_percentage_threshold(reference_id, reference_distance)
print("IDs within the 10% threshold:", ids_within_threshold)


# def calculate_toll_rate(df)->pd.DataFrame():
#     """
#     Calculate toll rates for each vehicle type based on the unrolled DataFrame.

#     Args:
#         df (pandas.DataFrame)

#     Returns:
#         pandas.DataFrame
#     """
#     # Wrie your logic here

#     return df


# def calculate_time_based_toll_rates(df)->pd.DataFrame():
#     """
#     Calculate time-based toll rates for different time intervals within a day.

#     Args:
#         df (pandas.DataFrame)

#     Returns:
#         pandas.DataFrame
#     """
#     # Write your logic here

#     return df
