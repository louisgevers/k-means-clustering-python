import numpy as np
import pandas as pd
from typing import Tuple

DATA_DIR = "../data/"

FILE_1D = "1d.txt"
FILE_2D = "2d.txt"

POINT_COLUMN = "point"
IDEAL_CLUSTER_COLUMN = "ideal cluster"

def read_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + filename, names=[POINT_COLUMN, IDEAL_CLUSTER_COLUMN])

def get_ideal_clusters(df: pd.DataFrame, parse=False) -> np.array:
    ideal_cluster_ids = df[IDEAL_CLUSTER_COLUMN].unique()
    ideal_clusters = list()
    for cluster_id in ideal_cluster_ids:
        cluster = df.loc[df[IDEAL_CLUSTER_COLUMN] == cluster_id]
        points = cluster[POINT_COLUMN]
        if parse:
            points = parse_2d_points(points)
        ideal_clusters.append(points.values)
    return np.array(ideal_clusters)

def get_1d_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Returns the points and ideal cluster for the 1D file"""
    df = read_dataset(FILE_1D)
    ideal_clusters = get_ideal_clusters(df)
    points = df[POINT_COLUMN].values
    points = points[:, np.newaxis] # wrap points in nested array for consistency with 2D
    return points, ideal_clusters

def parse_2d_points(point_column: pd.Series) -> pd.Series:
    """Parse [1.033, 3.23] string into float array"""
    return point_column.str[1:-1].str.split(expand=True).astype(float)

def get_2d_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Returns the points and ideal cluster for the 2D file"""
    df = read_dataset(FILE_2D)
    ideal_clusters = get_ideal_clusters(df, parse=True)
    parsed_points = parse_2d_points(df[POINT_COLUMN])
    points = parsed_points.values
    return points, ideal_clusters