
import os

import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

df_file_coordinate = pd.read_csv(config.get('COMMON', 'percorso_file_coordinate'), index_col=0)
assert 'Latitude' in df_file_coordinate.columns
assert 'Longitude' in df_file_coordinate.columns

print('\n\nDone')