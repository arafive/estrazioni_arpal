
import os
import ast
import cfgrib
import configparser

import numpy as np
import pandas as pd

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
    df_attributi = pd.DataFrame()
    
    for i, ds in enumerate(lista_ds):
        
        for v in [x for x in ds.data_vars]:
            if v not in dict_ds_variabili.keys():
                dict_ds_variabili[v] = [i]
            else:
                dict_ds_variabili[v].append(i)

            df_attributi = pd.concat([df_attributi, pd.DataFrame({'id_ds': i} | ds[v].attrs, index=[v])])
            
    ### Elimino le colonne i vuoi valori sono comuni a tutte le righe
    for i in df_attributi:
        if len(set(df_attributi[i].tolist())) == 1:
            df_attributi = df_attributi.drop(columns=[i])
                
    return dict_ds_variabili, df_attributi

# %%

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

    dict_ds_variabili, df_attributi = f_dizionario_ds_variabili(lista_ds)
    
    # TODO alla prossima commit -> aggiungi il ciclo sulle variabili
    
    # for v in ast.literal_eval(config.get('ECITA', 'variabili_da_estratte')):
    #     print(v)
        
        
        
    
    # sss

print('\n\nDone')
