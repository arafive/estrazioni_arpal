
import os
import sys
import logging
import collections
import configparser

import numpy as np
import pandas as pd

from datetime import timedelta

def f_logger(livello_minimo='INFO'):
    """Crea un logger per stampare log colorati sul terminale e normali su un file di log."""
    # https://betterstack.com/community/guides/logging/how-to-start-logging-with-python/
    
    levelname_completo = True # se è True allora verrà "info" oppure "INFO", altrimenti "I"
    numero_formato = 2 # 1, 2
    ### 1. --> INFO     | 2025-01-02 11:10:05.706 | studio_logging.py:146  | 1169383 >>> Informazioni generali su cosa sta facendo il programma.
    ### 2. --> 2025-01-02 11:07:14.359   info       studio_logging.py:146    1169027   Informazioni generali su cosa sta facendo il programma.
    
    resetta_colore = '\033[0m'
    dict_colori = {
        'DEBUG': '\033[3;36m',    # Ciano, 3; vuol dire italic
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[5;33m',  # Giallo, 5; vuol dire che lampeggia
        'ERROR': '\033[1;31m',    # Rosso, 1; vuol dire bold
        'CRITICAL': '\033[4;35m', # Magenta, 4; vuol dire sottolineato
        'D': '\033[3;36m',
        'I': '\033[32m',
        'W': '\033[5;33m',
        'E': '\033[1;31m',
        'C': '\033[4;35m'
    }
    
    class ColoredFormatter(logging.Formatter):
        def __init__(self, fmt, datefmt):
            super().__init__(fmt, datefmt)

        def format(self, record):
            record.filename = sys.argv[0].split('/')[-1] # In questo modo 'filename' è sempre il nome del __main__
            levelname = record.levelname
            
            # Se l'output è su un file, evita i colori
            if isinstance(levelname, str) and sys.stdout.isatty():
                log_color = dict_colori.get(levelname, resetta_colore)
                
                if levelname_completo:
                    record.levelname = f'{log_color}{levelname:<8}{resetta_colore}'
                else:
                    record.levelname = f'{log_color}{levelname[0:1]}{resetta_colore}'
            else:
                if levelname_completo:
                    record.levelname = f'{levelname:<8}'
                else:
                    record.levelname = levelname[0:1]

            if numero_formato == 2 and levelname_completo:
                record.levelname = record.levelname.lower()
                
            return super().format(record)
        
    handler = logging.StreamHandler(sys.stdout)

    if numero_formato == 1:
        formato = '%(levelname)s | %(asctime)s.%(msecs)03d | %(filename)s:%(lineno)-4s | %(process)d >>> %(message)s'
    elif numero_formato == 2:
        formato = '%(asctime)s.%(msecs)03d   %(levelname)s   %(filename)s:%(lineno)-4s   %(process)d   %(message)s'

    handler.setFormatter(ColoredFormatter(formato, datefmt='%Y-%m-%d %H:%M:%S'))
    
    logging.basicConfig(handlers=[handler])
    logger = logging.getLogger()
    level = logging.getLevelName(livello_minimo)
    logger.setLevel(level)
    
    return logger


config = configparser.ConfigParser()
config.read('./config.ini')
logger = f_logger(config.get('COMMON', 'livello_minimo_logging'))
del (config)
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #


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

    # print(str_output)
    logger.info(str_output)
    
    
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
    # print(f'Creata cartella {percorso_cartella}')
    logger.debug(f'Creata cartella {percorso_cartella}')

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
    
    msg = f'{int(secondi)}.{int(millisecondi)} sec'

    if int(minuti) > 0:
        msg = f'{minuti}:{secondi} min'
    
    if int(ore) > 0:
        msg = f'{ore}:{minuti}:{secondi} ore'
    
    if int(giorni) > 0:
        if int(giorni) == 1:
            msg = f'{giorni} giorno, {ore}:{minuti}:{secondi} ore'
        else:
            msg = f'{giorni} giorni, {ore}:{minuti}:{secondi} ore'
    
    if nota:
        msg = f'{nota}: {msg}'
        
    # print(msg)
    logger.debug(msg)
    

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
            
    ### Elimino le colonne i cui valori sono comuni a tutte le righe
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
    

def f_check_duplicati(df):
    """Controlla se nel dataframe ci sono dei duplicati.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame

    """
    indici = [str(x) for x in df.index.tolist()]
    duplicati = [x for x, conteggio in collections.Counter(indici).items() if conteggio > 1]
    
    if len(duplicati) > 0:
        # logger.warning(f'Ho trovato {len(duplicati)}: {duplicati}')
        logger.warning(f'Ho trovato {len(duplicati)} duplicati')
