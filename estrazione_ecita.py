
import os
import ast
import cfgrib
import configparser

import numpy as np
import pandas as pd

def f_crea_cartella(percorso_cartella):
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
    print(f'Creata cartella {percorso_cartella}\n')

    return percorso_cartella


def f_dizionario_ds_variabili(lista_ds):
    """Ritorna il dizionario che lega dataset alle variabili.
    
    Parameters
    ----------
    lista_ds : list
        Lista che contiene i vari dataset.

    Returns
    -------
    dict_ds_variabili : dict
        Dizionario dove le chiavi sono le variabili e
        i valori delle liste che contengono gli indici
        dei dataset in 'lista_ds' che contengono
        quella variabile.
    df_attributi : pandas.core.frame.DataFrame
        Il dataframe che contiene gli attributi, oltre
        all'indice del dataset che contiene quella variabile.

    """
    dict_ds_variabili = {}
    df_attrs = pd.DataFrame()
    
    for i, ds in enumerate(lista_ds):
        
        for v in [x for x in ds.data_vars]:
            if v not in dict_ds_variabili.keys():
                dict_ds_variabili[v] = [i]
            else:
                dict_ds_variabili[v].append(i)

            # df_attrs = pd.concat([df_attrs, pd.DataFrame({'id_ds': i} | ds[v].attrs, index=[v])]) # solo per python >= 3.9
            df_attrs = pd.concat([df_attrs, pd.DataFrame({**{'id_ds': i},  **ds[v].attrs}, index=[v])])
            
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

    return dict_ds_variabili, df_attrs

# %%

config = configparser.ConfigParser()
config.read('./config.ini')

cartella_madre_estrazione = f_crea_cartella(config.get('COMMON', 'cartella_madre_estrazione'))

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

    dict_ds_variabili, df_attrs = f_dizionario_ds_variabili(lista_ds)
    sss
    for v in ast.literal_eval(config.get('ECITA', 'variabili_da_estratte')):
        df_sub_attrs = df_attrs.loc[v]
        # TODO alla prossima commit -> cicla correttamente sulla tripletta per creare le cartelle
        
        # for i in df_sub_attrs.index:
            # print(i)

        # print(v, df_attrs.loc[v, 'GRIB_dataType'], df_attrs.loc[v, 'GRIB_typeOfLevel'])
        # print()

        # sss
        
    

print('\n\nDone')
