
import os
import ast
import time
import string
import cfgrib
import configparser

import numpy as np
import pandas as pd

from joblib import delayed
from joblib import Parallel

from funzioni import f_log_ciclo_for
from funzioni import f_crea_cartella
from funzioni import f_printa_tempo_trascorso
from funzioni import f_dataframe_ds_variabili
from funzioni import f_round
    

config = configparser.ConfigParser()
config.read('./config.ini')

cartella_madre_estrazione = f_crea_cartella(f"{config.get('COMMON', 'cartella_madre_estrazione')}/ECMWF")

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

    percorso_file_grib = f"{config.get('ECITA', 'percorso_cartella_grib')}/{sub_cartella_grib}"
    nome_file_grib = f"ecmf_0.1_{d.year}{d.month:02d}{d.day:02d}{config.get('COMMON', 'ora_start_forecast')}_181x161_2_20_34_50_undef_undef.grb"

    if not os.path.exists(f'{percorso_file_grib}/{nome_file_grib}'):
        print(f'!!! File {nome_file_grib} non presente nella cartella {percorso_file_grib}. Continuo')
        return
        # continue
        
    lista_ds = cfgrib.open_datasets(f'{percorso_file_grib}/{nome_file_grib}',
                                    backend_kwargs={'indexpath': f'/tmp/{nome_file_grib}_lista_ds.idx'})
    
    # global df_attrs
    df_attrs = f_dataframe_ds_variabili(lista_ds)
    
    ### Ciclo sulle variabili
    for v in ast.literal_eval(config.get('ECITA', 'variabili_da_estratte')):
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
            #                   [f'Variabile (indice {i}) ', v, ast.literal_eval(config.get('ECITA', 'variabili_da_estratte'))]])
            
            nome_var = df_sub_attrs.index[0]
            grib_dataType = df_sub_attrs.iloc[i]['GRIB_dataType']
            grib_typeOfLevel = df_sub_attrs.iloc[i]['GRIB_typeOfLevel']

            ds = lista_ds[df_sub_attrs.iloc[i]['id_ds']]
            
            inizio_run = pd.to_datetime(ds['time'].values)
            tempi = pd.to_datetime(ds['valid_time'].values) # equivalente (ma più robusto) di "pd.to_datetime([ds['time'].values + x for x in ds['step'].values])"
            lon_2D, lat_2D = np.meshgrid(ds['longitude'], ds['latitude'])
            
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

                    if grib_dataType == 'an' and grib_typeOfLevel in ['surface', 'potentialVorticity'] and len(ds[nome_var].values.shape) == 2:
                        ### (latitudini, longitudini)
                        estrazione = ds[nome_var].values[lat_min, lon_min]
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[tempi], columns=[lettera])], axis=1)

                    elif grib_dataType == 'an' and grib_typeOfLevel == 'isobaricInhPa' and len(ds[nome_var].values.shape) == 3:
                        ### (livelli, latitudini, longitudini)
                        estrazione = ds[nome_var].values[:, lat_min, lon_min].squeeze()
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[livelli], columns=[lettera])], axis=1)

                    elif grib_dataType == 'fc' and grib_typeOfLevel == 'potentialVorticity' and len(ds[nome_var].values.shape) == 2:
                        ### (latitudini, longitudini)
                        estrazione = ds[nome_var].values[lat_min, lon_min]
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[tempi], columns=[lettera])], axis=1)

                    elif grib_dataType == 'fc' and grib_typeOfLevel in ['surface', 'potentialVorticity'] and len(ds[nome_var].values.shape) == 3:
                        ### (tempi, latitudini, longitudini)
                        estrazione = ds[nome_var].values[:, lat_min, lon_min].squeeze()
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[tempi], columns=[lettera])], axis=1)

                    elif grib_dataType == 'fc' and grib_typeOfLevel == 'isobaricInhPa' and len(ds[nome_var].values.shape) == 4:
                        ### (tempi, livelli, latitudini, longitudini)

                        df_tmp = pd.DataFrame()
                        
                        for ind_l, l in enumerate(livelli):
                            estrazione_tempi = ds[nome_var].values[:, ind_l, lat_min, lon_min].squeeze()
                            df_tmp = pd.concat([df_tmp, pd.DataFrame(estrazione_tempi, index=[tempi], columns=[f'{lettera}_{l}'])], axis=1)

                        df_estrazione = pd.concat([df_estrazione, df_tmp], axis=1)
                        del (df_tmp)

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
    Parallel(n_jobs=int(config.get('COMMON', 'job')), verbose=1000)(delayed(f_estrazione)(d) for d in lista_date_start_forecast)
    
print('\n\nDone')
