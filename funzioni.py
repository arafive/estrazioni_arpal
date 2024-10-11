
import os

import numpy as np
import pandas as pd

from datetime import timedelta


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
    
