
import os
import ast
import time
import string
import cfgrib
import configparser

import numpy as np
import pandas as pd
import xarray as xr

from datetime import timedelta

from joblib import delayed
from joblib import Parallel

def f_log_ciclo_for(lista_di_liste):
    """Log per un ciclo for.
    
    Parameters
    ----------
    lista_di_liste : list
        Lista_di_liste. Ogni lista contiene 3 elementi:
        I   la descrizione
        II  l'elemento di ogni ciclo
        III la lista iterata
        
    """
    str_output = ''
    for n, i in enumerate(lista_di_liste, 1):
        assert len(i) == 3, 'Ci sono meno di 3 elementi. Modifica.'
        
        if not type(i[2]) == list:
            i[2] = list(i[2])
            
        sub_str = f'{i[0]}{i[1]} [{i[2].index(i[1]) + 1}/{len(i[2])}]'
        str_output = str_output + sub_str
        if not n == len(lista_di_liste):
            str_output = str_output + ' · '

    print(str_output)
    
    
def f_crea_cartella(percorso_cartella, print_messaggio=True):
    """Crea una cartella, printa per conferma e ritorna il percorso.
    
    Parameters
    ----------
    percorso_cartella : str
        Lista che contiene i vari dataset.

    Returns
    -------
    percorso_cartella : str
        Lista che contiene i vari dataset.

    """
    os.makedirs(percorso_cartella, exist_ok=True)
    if print_messaggio:
        print(f'Creata cartella {percorso_cartella}\n')

    return percorso_cartella


def f_printa_tempo_trascorso(t_inizio, t_fine, nota=False):
    """Printa il tempo trascorso.
    
    Parameters
    ----------
    t_inizio : float
        Tempo iniziale generato con time.time().
    t_fine : float
        Tempo finale generato con time.time().

    """
    elapsed_tempo = timedelta(seconds=t_fine - t_inizio)
    
    giorni = f'{elapsed_tempo.days:01}'
    ore = f'{elapsed_tempo.seconds//3600:02}'
    minuti = f'{elapsed_tempo.seconds//60%60:02}'
    secondi = f'{elapsed_tempo.seconds%60:02}'
    millisecondi = elapsed_tempo.microseconds / 1000
    
    msg = f'\n>>> {int(secondi)}.{int(millisecondi)} sec'

    if int(minuti) > 0:
        msg = f'\n>>> {minuti}:{secondi} min'
    
    if int(ore) > 0:
        msg = f'\n>>> {ore}:{minuti}:{secondi} ore'
    
    if int(giorni) > 0:
        if int(giorni) == 1:
            msg = f'\n>>> {giorni} giorno, {ore}:{minuti}:{secondi} ore'
        else:
            msg = f'\n>>> {giorni} giorni, {ore}:{minuti}:{secondi} ore'
    
    if nota:
        msg = msg[:4] + f' {nota}: ' + msg[5:]
        
    print(msg)
    

def f_dataframe_ds_variabili(lista_ds):
    """Ritorna il dataframe che lega dataset alle variabili.
    
    Parameters
    ----------
    lista_ds : list
        Lista che contiene i vari dataset.

    Returns
    -------
    df_attributi : pandas.core.frame.DataFrame
        Il dataframe che contiene gli attributi, oltre
        all'indice del dataset che contiene quella variabile.

    """
    df_attrs = pd.DataFrame()
    
    for i, ds in enumerate(lista_ds):
        for v in [x for x in ds.data_vars]:
            df_attrs = pd.concat([df_attrs, pd.DataFrame({**{'id_ds': i}, **ds[v].attrs}, index=[v])])
            
    ### Elimino le colonne i vuoi valori sono comuni a tutte le righe
    for i in df_attrs:
        if len(set(df_attrs[i].tolist())) == 1:
            df_attrs = df_attrs.drop(columns=[i])
    
    df_attrs = df_attrs.drop(columns=['long_name']) # doppione di 'GRIB_name'
    df_attrs = df_attrs.drop(columns=['standard_name']) # doppione di 'GRIB_cfName'
    df_attrs = df_attrs.drop(columns=['GRIB_cfName']) # non ha importanza, sono quasi tutti 'unknown'
    df_attrs = df_attrs.drop(columns=['units']) # doppione di 'GRIB_units'
    df_attrs = df_attrs.drop(columns=['GRIB_shortName', 'GRIB_cfVarName']) # doppioni dell'index
    
    if 'GRIB_dataType' not in df_attrs:
        df_attrs['GRIB_dataType'] = 'fc' # Nei grib più recenti 'an' e 'fc' sono uniti

    return df_attrs


def f_round(a, digits=3):
    """Arrotonda i float.

    Parameters
    ----------
    a : float
        Un numero.
    digits : int, optional
        Quante cifre dopo la virgola mantenere. The default is 3.

    Returns
    -------
    np.float32

    """
    if str(a).split('.')[0] == '0':
        return np.float32(a)
    
    else:
        return np.float32(np.round(a, decimals=digits))

# %%


config = configparser.ConfigParser()
config.read('./config.ini')

cartella_madre_estrazione = f_crea_cartella(f"{config.get('COMMON', 'cartella_madre_estrazione')}/BOLAM")

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

    percorso_file_grib = f"{config.get('BOLAM', 'percorso_cartella_grib')}/{sub_cartella_grib}"
    nome_file_grib = f"bo08_{d.year}{d.month:02d}{d.day:02d}{config.get('COMMON', 'ora_start_forecast')}.grib2"

    if not os.path.exists(f'{percorso_file_grib}/{nome_file_grib}'):
        print(f'!!! File {nome_file_grib} non presente nella cartella {percorso_file_grib}. Continuo')
        return
        # continue
    
    lista_ds = cfgrib.open_datasets(f'{percorso_file_grib}/{nome_file_grib}',
                                    # backend_kwargs={'indexpath': ''})
                                    # backend_kwargs={'indexpath': None})
                                    backend_kwargs={'indexpath': f'/tmp/{nome_file_grib}.idx'})
    
    # global df_attrs
    df_attrs = f_dataframe_ds_variabili(lista_ds)
    df_attrs = df_attrs.drop('unknown', axis=0)
    
    ds_tp3 = xr.open_dataset(f'{percorso_file_grib}/{nome_file_grib}', engine='cfgrib',
                             filter_by_keys={'discipline': 0, 'parameterNumber': 8, 'parameterCategory': 1},
                             backend_kwargs={'indexpath': ''})
    ds_tp3 = ds_tp3.rename({'unknown': 'tp3'})
    
    ds_cp3 = xr.open_dataset(f'{percorso_file_grib}/{nome_file_grib}', engine='cfgrib',
                             filter_by_keys={'discipline': 0, 'parameterNumber': 10, 'parameterCategory': 1},
                             backend_kwargs={'indexpath': ''})
    ds_cp3 = ds_cp3.rename({'acpcp': 'cp3'})
    
    ds_sf3 = xr.open_dataset(f'{percorso_file_grib}/{nome_file_grib}', engine='cfgrib',
                             filter_by_keys={'discipline': 0, 'parameterNumber': 29, 'parameterCategory': 1},
                             backend_kwargs={'indexpath': ''})
    ds_sf3 = ds_sf3.rename({'unknown': 'sf3'})

    lista_ds.append(ds_cp3)
    df_attrs = df_attrs.rename(index={'acpcp': 'cp3'})
    df_attrs.loc['cp3', 'id_ds'] = int(df_attrs['id_ds'].max()) + 1
    
    lista_ds.append(ds_tp3)
    df_tp3 = pd.DataFrame('unknown', index=['tp3'], columns=df_attrs.columns)
    df_tp3.loc['tp3', 'id_ds'] = int(df_attrs['id_ds'].max()) + 1
    df_tp3.loc['tp3', 'GRIB_typeOfLevel'] = 'surface'
    df_tp3.loc['tp3', 'GRIB_stepType'] = 'accum'
    df_tp3.loc['tp3', 'GRIB_name'] = 'Total precipitation'
    df_tp3.loc['tp3', 'GRIB_units'] = 'kg m**-2'
    df_tp3.loc['tp3', 'GRIB_dataType'] = 'fc'
    df_attrs = pd.concat([df_attrs, df_tp3], axis=0)

    lista_ds.append(ds_sf3)
    df_sf3 = pd.DataFrame('unknown', index=['sf3'], columns=df_attrs.columns)
    df_sf3.loc['sf3', 'id_ds'] = int(df_attrs['id_ds'].max()) + 1
    df_sf3.loc['sf3', 'GRIB_typeOfLevel'] = 'surface'
    df_sf3.loc['sf3', 'GRIB_stepType'] = 'accum'
    df_sf3.loc['sf3', 'GRIB_name'] = 'Total snowfall'
    df_sf3.loc['sf3', 'GRIB_units'] = 'm'
    df_sf3.loc['sf3', 'GRIB_dataType'] = 'fc'
    df_attrs = pd.concat([df_attrs, df_sf3], axis=0)
    
    ### Ciclo sulle variabili
    for v in ast.literal_eval(config.get('BOLAM', 'variabili_da_estratte')):
        
        if v not in df_attrs.index:
            print(f'!!! Variabile {v} non presente nel file {nome_file_grib}. Continuo')
            continue
        
        df_sub_attrs = df_attrs.loc[v, :]
        
        if type(df_sub_attrs) == pd.core.series.Series:
            df_sub_attrs = df_sub_attrs.to_frame().T

        ### Ciclo sulla posizione degli indici
        for i in range(df_sub_attrs.shape[0]):
            t_inizio_v = time.time()
            
            # f_log_ciclo_for([['Data ', d, lista_date_start_forecast],
            #                   [f'Variabile (indice {i}) ', v, ast.literal_eval(config.get('BOLAM', 'variabili_da_estratte'))]])
            
            nome_var = df_sub_attrs.index[0]
            grib_dataType = df_sub_attrs.iloc[i]['GRIB_dataType']
            grib_typeOfLevel = df_sub_attrs.iloc[i]['GRIB_typeOfLevel']
            
            ds = lista_ds[df_sub_attrs.iloc[i]['id_ds']]
            # sss
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

                    var_np_ruotata = np.rot90(ds[nome_var].values.T, 1)
                    
                    ### Per il BOLAM c'è solo 'fc'
                    if grib_typeOfLevel in ['surface', 'meanSea', 'heightAboveGround'] and len(ds[nome_var].values.shape) == 3:
                        ### (tempi, latitudini, longitudini) -> (latitudini, longitudini, tempi)
                        estrazione = var_np_ruotata[lat_min, lon_min, :].squeeze()
                        df_estrazione = pd.concat([df_estrazione, pd.DataFrame(estrazione, index=[tempi], columns=[lettera])], axis=1)

                    elif grib_typeOfLevel == 'isobaricInhPa' and len(ds[nome_var].values.shape) == 4:
                        ### (tempi, livelli, latitudini, longitudini) -> (latitudini, longitudini, livelli, tempi)

                        df_tmp = pd.DataFrame()
                        
                        for ind_l, l in enumerate(livelli):
                            estrazione_tempi = var_np_ruotata[lat_min, lon_min, ind_l, :].squeeze()
                            df_tmp = pd.concat([df_tmp, pd.DataFrame(estrazione_tempi, index=[tempi], columns=[f'{lettera}_{l}'])], axis=1)

                        df_estrazione = pd.concat([df_estrazione, df_tmp], axis=1)
                        del (df_tmp)

                    else:
                        raise Exception('Caso non contemplato: ', nome_var, grib_dataType, grib_typeOfLevel, ds[nome_var].values.shape, len(ds[nome_var].values.shape))

                df_estrazione = df_estrazione.astype(float).map(f_round, digits=3)
                df_estrazione.to_csv(f"{cartella_estrazione}/{str(inizio_run).split(' ')[0]}.csv", index=True, header=True, mode='w', na_rep=np.nan)
                
            f_printa_tempo_trascorso(t_inizio_v, time.time(), nota=f'Tempo per variabile {v}')
                
    f_printa_tempo_trascorso(t_inizio_d, time.time(), nota=f'Tempo per d = {d}')
    print()
    
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #


if int(config.get('COMMON', 'job_joblib')) == 0:
    ### Ciclo sulle date
    for d in lista_date_start_forecast:
        f_estrazione(d)
    
else:
    Parallel(n_jobs=int(config.get('COMMON', 'job_joblib')), verbose=1000)(delayed(f_estrazione)(d) for d in lista_date_start_forecast)
    
print('\n\nDone')
