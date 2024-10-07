
import os
import cfgrib
import configparser

import numpy as np
import pandas as pd


config = configparser.ConfigParser()
config.read('./config.ini')

df_file_coordinate = pd.read_csv(config.get('COMMON', 'percorso_file_coordinate'), index_col=0)
assert 'Latitude' in df_file_coordinate.columns
assert 'Longitude' in df_file_coordinate.columns

df_file_coordinate = df_file_coordinate[['Latitude', 'Longitude']]

lista_date_start_forecast = pd.date_range(f"{config.get('COMMON', 'data_inizio_estrazione')} {config.get('COMMON', 'ora_start_forecast')}:00:00",
                                          f"{config.get('COMMON', 'data_fine_estrazione')} {config.get('COMMON', 'ora_start_forecast')}:00:00",
                                          freq='1D')

### Ciclo sulle date
for d in lista_date_start_forecast:
    sub_cartella_grib = f'{d.year}/{d.month:02d}/{d.day:02d}'
    print(sub_cartella_grib)

    percorso_file_grib = f"{config.get('ECITA', 'percorso_cartella_grib')}/{sub_cartella_grib}"
    nome_file_grib = f"ecmf_0.1_{d.year}{d.month:02d}{d.day:02d}{config.get('COMMON', 'ora_start_forecast')}_181x161_2_20_34_50_undef_undef.grb"

    lista_ds = cfgrib.open_datasets(f'{percorso_file_grib}/{nome_file_grib}',
                                    indexpath=f'/tmp/{nome_file_grib}.idx')

print('\n\nDone')