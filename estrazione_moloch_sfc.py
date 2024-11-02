
import os
import ast
import time
import string
import cfgrib
import configparser
import multiprocessing

import numpy as np
import pandas as pd
import xarray as xr

from joblib import delayed
from joblib import Parallel

from funzioni import f_log_ciclo_for
from funzioni import f_crea_cartella
from funzioni import f_printa_tempo_trascorso
from funzioni import f_dataframe_ds_variabili
from funzioni import f_round


config = configparser.ConfigParser()
config.read('./config.ini')

cartella_madre_estrazione = f_crea_cartella(f"{config.get('COMMON', 'cartella_madre_estrazione')}/MOLOCHsfc")

df_file_coordinate = pd.read_csv(config.get('COMMON', 'percorso_file_coordinate'), index_col=0)
assert 'Latitude' in df_file_coordinate.columns
assert 'Longitude' in df_file_coordinate.columns

df_file_coordinate = df_file_coordinate[['Latitude', 'Longitude']]

lista_date_start_forecast = pd.date_range(f"{config.get('COMMON', 'data_inizio_estrazione')} {config.get('COMMON', 'ora_start_forecast')}:00:00",
                                          f"{config.get('COMMON', 'data_fine_estrazione')} {config.get('COMMON', 'ora_start_forecast')}:00:00",
                                          freq='1D')


def f_estrazione(d):
# for d in lista_date_start_forecast:
    t_inizio_d = time.time()
    f_log_ciclo_for([['Data ', d, lista_date_start_forecast]])
    
    sub_cartella_grib = f'{d.year}/{d.month:02d}/{d.day:02d}'

    percorso_file_grib = f"{config.get('MOLOCHsfc', 'percorso_cartella_grib')}/{sub_cartella_grib}"
    nome_file_grib = f"molita15sfc_{d.year}{d.month:02d}{d.day:02d}{config.get('COMMON', 'ora_start_forecast')}.grib2"

    if not os.path.exists(f'{percorso_file_grib}/{nome_file_grib}'):
        print(f'!!! File {nome_file_grib} non presente nella cartella {percorso_file_grib}. Continuo')
        return
        # continue
    
    lista_ds = cfgrib.open_datasets(f'{percorso_file_grib}/{nome_file_grib}',
                                    backend_kwargs={'indexpath': f'/tmp/{nome_file_grib}_lista_ds.idx'})

    # global df_attrs
    df_attrs = f_dataframe_ds_variabili(lista_ds)
    
    ds_tp1 = xr.open_dataset(f'{percorso_file_grib}/{nome_file_grib}', engine='cfgrib',
                             filter_by_keys={'discipline': 0, 'parameterCategory': 1, 'parameterNumber': 8},
                             backend_kwargs={'indexpath': f'/tmp/{nome_file_grib}_tp1.idx'})
    ds_tp1 = ds_tp1.rename({'unknown': 'tp1'})

    lista_ds.append(ds_tp1)
    df_tp1 = pd.DataFrame('unknown', index=['tp1'], columns=df_attrs.columns)
    df_tp1.loc['tp1', 'id_ds'] = int(df_attrs['id_ds'].max()) + 1
    df_tp1.loc['tp1', 'GRIB_typeOfLevel'] = 'surface'
    df_tp1.loc['tp1', 'GRIB_stepType'] = 'accum'
    df_tp1.loc['tp1', 'GRIB_name'] = 'Total precipitation'
    df_tp1.loc['tp1', 'GRIB_units'] = 'kg m**-2'
    df_tp1.loc['tp1', 'GRIB_dataType'] = 'fc'
    df_attrs = pd.concat([df_attrs, df_tp1], axis=0)

    df_attrs = df_attrs.drop('unknown', axis=0) # deve stare in fondo

    ### Ciclo sulle variabili
    for v in ast.literal_eval(config.get('MOLOCHsfc', 'variabili_da_estratte')):
        t_inizio_v = time.time()
        
        if v not in df_attrs.index:
            print(f'!!! Variabile {v} non presente nel file {nome_file_grib}. Continuo')
            continue
        
        df_sub_attrs = df_attrs.loc[v, :]
        
        if type(df_sub_attrs) == pd.core.series.Series:
            df_sub_attrs = df_sub_attrs.to_frame().T

        ### Ciclo sulla posizione degli indici
        for i in range(df_sub_attrs.shape[0]):
            
            # f_log_ciclo_for([['Data ', d, lista_date_start_forecast],
            #                   [f'Variabile (indice {i}) ', v, ast.literal_eval(config.get('MOLOCHsfc', 'variabili_da_estratte'))]])

            nome_var = df_sub_attrs.index[0]
            grib_dataType = df_sub_attrs.iloc[i]['GRIB_dataType']
            grib_typeOfLevel = df_sub_attrs.iloc[i]['GRIB_typeOfLevel']
            
            ds = lista_ds[df_sub_attrs.iloc[i]['id_ds']]

            inizio_run = pd.to_datetime(ds['time'].values)
            tempi = pd.to_datetime(ds['valid_time'].values) # equivalente (ma più robusto) di "pd.to_datetime([ds['time'].values + x for x in ds['step'].values])"
            lon_2D, lat_2D = ds['longitude'].values, ds['latitude'].values
            lat_2D, lon_2D = np.rot90(lat_2D.T, 1), np.rot90(lon_2D.T, 1)
            
            if ds[grib_typeOfLevel].values.shape == ():
                livelli = np.array([ds[grib_typeOfLevel].values])
            else:
                livelli = ds[grib_typeOfLevel].values
                
            livelli = [int(x) for x in livelli]

            ### Ciclo sulle stazioni
            for s in df_file_coordinate.index:
                cartella_estrazione = f_crea_cartella(f"{cartella_madre_estrazione}/{config.get('COMMON', 'ora_start_forecast')}/{nome_var}/{grib_dataType}/{grib_typeOfLevel}/{s}", print_messaggio=False)
                
                lat_s = df_file_coordinate.loc[s, 'Latitude']
                lon_s = df_file_coordinate.loc[s, 'Longitude']
                
                distanze_2D = (np.abs(lon_2D - lon_s) + np.abs(lat_2D - lat_s))
                distanze_1D = np.sort(distanze_2D.flatten())
                
                df_estrazione = pd.DataFrame()
                
                if os.path.exists(f"{cartella_estrazione}/{str(inizio_run).split(' ')[0]}.csv"):
                    # print(f"{cartella_estrazione}/{str(inizio_run).split(' ')[0]}.csv esiste. Continuo." )
                    continue

                ### Ciclo sui punti
                for p, lettera, dist in zip(range(int(config.get('COMMON', 'punti_piu_vicini_da_estrarre'))), list(string.ascii_uppercase), distanze_1D):
                    lat_min, lon_min = np.where(distanze_2D == dist)
                    
                    # !!! Possono esserci più di un punto con la stessa distanza dalla stazione
                    if lat_min.shape[0] > 1:
                        lat_min = lat_min[0]
                    if lon_min.shape[0] > 1:
                        lon_min = lon_min[0]

                    var_np_ruotata = np.rot90(ds[nome_var].values.T, 1)

                    ### Per il MOLOCHsfc c'è solo 'fc'
                    if grib_typeOfLevel in ['surface', 'heightAboveGround'] and len(ds[nome_var].values.shape) == 3:
                        ### if testato solo per ['t2m', 'v10', 'u10', 'tp1']
                        ### (tempi, latitudini, longitudini) -> (latitudini, longitudini, tempi)
                        estrazione = var_np_ruotata[lat_min, lon_min, :].squeeze()
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[tempi], columns=[lettera])], axis=1)

                    else:
                        raise Exception('Caso non contemplato: ', nome_var, grib_dataType, grib_typeOfLevel, ds[nome_var].values.shape, len(ds[nome_var].values.shape))

                try:
                    df_estrazione = df_estrazione.astype(float).map(f_round, digits=3)
                except AttributeError:
                    df_estrazione = df_estrazione.astype(float).applymap(f_round, digits=3)
                    
                df_estrazione.to_csv(f"{cartella_estrazione}/{str(inizio_run).split(' ')[0]}.csv", index=True, header=True, mode='w', na_rep=np.nan)
                
            f_printa_tempo_trascorso(t_inizio_v, time.time(), nota=f'Tempo per variabile {v} (indice {i})')
                
    f_printa_tempo_trascorso(t_inizio_d, time.time(), nota=f'Tempo per d = {d}')
    print()

# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #


if int(config.get('COMMON', 'job')) == 0:
    ### Ciclo sulle date
    for d in lista_date_start_forecast:
        f_estrazione(d)

else:
    if config.get('COMMON', 'tipo_di_parallellizzazione') == 'joblib':
        Parallel(n_jobs=int(config.get('COMMON', 'job')), verbose=1000)(delayed(f_estrazione)(d) for d in lista_date_start_forecast)
    
    elif config.get('COMMON', 'tipo_di_parallellizzazione') == 'multiprocessing':
        pool = multiprocessing.Pool(processes=int(config.get('COMMON', 'job')))
        pool.map(f_estrazione, lista_date_start_forecast)
        pool.close()
        pool.join() # Aspetta che tutti finiscano
        
print('\n\nDone')
