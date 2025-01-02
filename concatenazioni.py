
import os
import time
# import string
import configparser
import multiprocessing

import numpy as np
import pandas as pd

from metpy.units import units
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.calc import relative_humidity_from_dewpoint
from metpy.calc import dewpoint_from_relative_humidity
from metpy.calc import potential_temperature
from metpy.calc import wind_direction
from metpy.calc import virtual_temperature

from joblib import delayed
from joblib import Parallel

# lista_possibili_cartelle_lavoro = [
#     '/media/daniele/Daniele2TB/repo/estrazioni_arpal',
#     '/run/media/daniele.carnevale/Daniele2TB/repo/estrazioni_arpal',
#     ]

# os.chdir([x for x in lista_possibili_cartelle_lavoro if os.path.exists(x)][0])
# del (lista_possibili_cartelle_lavoro)

from funzioni import f_log_ciclo_for
from funzioni import f_crea_cartella
from funzioni import f_round
from funzioni import f_printa_tempo_trascorso
from funzioni import f_logger
from funzioni import f_check_duplicati

config = configparser.ConfigParser()
config.read('./config.ini')

logger = f_logger(config.get('COMMON', 'livello_minimo_logging'))

dict_config_modelli = {
    'ECITA': 'ECMWF',
    'ECMWF': 'ECMWF',
    'BOLAM': 'BOLAM',
    'MOLOCH': 'MOLOCH',
    'MOLOCHsfc': 'MOLOCHsfc',
}

cartella_madre_estrazione = f"{config.get('COMMON', 'cartella_madre_estrazione')}/{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}"
ora_start_forecast = f"{config.get('COMMON', 'ora_start_forecast')}"
punti_piu_vicini_da_estrarre = int(f"{config.get('COMMON', 'punti_piu_vicini_da_estrarre')}")

# cartella_dati_osservati = f"{config.get('CONCATENAZIONI', 'cartella_dati_osservati')}"
cartella_madre_output_concatenazioni = f"{config.get('CONCATENAZIONI', 'cartella_madre_output_concatenazioni')}"
range_previsionale = f"{config.get('CONCATENAZIONI', 'range_previsionale')}"

cartella_output_concatenazioni = f_crea_cartella(f"{cartella_madre_output_concatenazioni}/{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}/{ora_start_forecast}/{range_previsionale}")
cartella_tmp = f_crea_cartella(f'{cartella_output_concatenazioni}/tmp')

# %%

lista_variabili = sorted([x for x in os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}') if not x.endswith('.txt')])

def f_concatenazione(v):
# for v in lista_variabili:
    
    # if v == 'w' and config.get('CONCATENAZIONI', 'modello') == 'ECMWF':
    #     logger.info('Modello ECMWF, salto la variabile w perchè non ha i livelli in quota, oltre 500 hPa')
    #     continue
        
    t_inizio_v = time.time()
    
    nome_df_finale = f"{cartella_tmp}/df_{v}_{range_previsionale}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv"
    
    if os.path.exists(nome_df_finale):
        logger.info(f'{nome_df_finale} esiste già. Continuo.')
        # continue
        return
        
    df_v = pd.DataFrame()

    # lista_cartelle_an_fc = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}'))
    # for f in lista_cartelle_an_fc:
    f = 'fc'  # Non prendo l'analisi

    lista_livelli = [x for x in sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}')) if not '.' in x]

    for l in lista_livelli:
        if l == 'potentialVorticity':
            v_nome = f'{v}2PVU'
        else:
            v_nome = v

        global lista_stazioni
        lista_stazioni = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}'))

        for s in lista_stazioni:
        # for s in lista_stazioni[0:2]:
            # t_inizio_s = time.time()
            f_log_ciclo_for([[f'Variabile ({l}) ', v, lista_variabili], ['Stazione ', s, lista_stazioni]])

            df_s = pd.DataFrame()

            lista_file_tempi = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}/{s}'))
            lista_datetime = pd.to_datetime([x.split('.')[0] for x in lista_file_tempi])
            
            # lista_datetime = lista_datetime[0:10]
            
            for t, d in zip(lista_file_tempi, lista_datetime):
                # print(v, f, l, s, t)
                cartella_df = f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}/{s}'

                try:
                    df = pd.read_csv(f'{cartella_df}/{t}', index_col=0, parse_dates=True)
                except pd.errors.EmptyDataError as e:
                    ### ECMWF/00/cape/fc/surface/TESTI/2021-02-03.csv erano vuoti, non so perchè
                    logger.warning(e)
                    continue

                if '.' in df.columns[0]:
                    ### Colpa mia, solo nell'ecita
                    logger.debug('df contiene un punto (.) .')
                    df.columns = [x.split('.')[0] for x in df.columns]
                    df.columns = [f"{x.split('_')[0]}_{x.split('_')[1]}" for x in df.columns]
                
                if len(df.columns[0].split('_')) == 2 and dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':
                    ### Ecita non ha sempre avuto un numero costante di livelli in pressione.
                    ### Devo tenere solo quelli sempre presenti.
                    ### I venti tipo u10 non vengono tolti.
                    logger.debug('Tolgo dei livelli dal df di ECITA.')
                    livelli_hPa_da_togliere = [10] + [int(x) for x in np.arange(25, 325, 25)]
                    for livello_da_togliere in livelli_hPa_da_togliere:
                        df = df.drop(columns=[x for x in df.columns if str(livello_da_togliere) in x.split('_')])
                    
                if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF' and v in ['tp', 'cp']:
                    ### Ecita: non ha la tp3 ma una precipitazione cumulata dallo start fino alla fine
                    logger.debug(f'Rendo la {v} di ECITA una cumulata nelle ore.')
                    df_shift = df.shift(periods=1, fill_value=0) # periods=1 per la tp3
                    df = df - df_shift

                if df.index[0].hour == 0:
                    ### Ecita: dal 2023-01-14 l'analisi e il forecast non sono più separati. Devo togliere la prima riga
                    logger.debug('Tolgo la prima riga dal df di ECITA.')
                    df = df.drop(df.index[0], axis=0)

                freq = '1h' if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'MOLOCHsfc' else '3h'
                df = df.loc[df.index.intersection(pd.date_range(d + pd.DateOffset(hours=int(range_previsionale.split('-')[0])),
                                                                d + pd.DateOffset(hours=int(range_previsionale.split('-')[1])),
                                                                freq=freq))]

                v_nome = v_nome.replace('_', '')  # Nel caso di 'cape_con' -> 'capecon'
                df.columns = [f'{v_nome}_{x}_{s}' for x in df.columns]

                df_s = pd.concat([df_s, df], axis=0)
                
            f_check_duplicati(df_s)
            df_s = df_s[~df_s.index.duplicated(keep='first')]
            
            try:
                df_v = pd.concat([df_v, df_s], axis=1)
            except pd.errors.InvalidIndexError as e:
                ### Problemi con ECMWF/00/cape/fc/surface/TESTI/2023-12-31.csv
                logger.warning(e)
                continue
            
    lista_completa_datetime = pd.date_range(lista_datetime[0], lista_datetime[-1] + pd.DateOffset(hours=24), freq=freq)

    ### Per non rischiare di generare un mostro di .csv, devo salvare a pezzetti.

    f_check_duplicati(df_v)
    df_v = df_v[~df_v.index.duplicated(keep='last')]
    
    df_v = df_v.reindex(lista_completa_datetime, fill_value=np.nan)
    df_v = df_v.sort_index()
    df_v = df_v.dropna()

    df_v.to_csv(nome_df_finale, index=True, header=True, mode='w', na_rep=np.nan)

    f_printa_tempo_trascorso(t_inizio_v, time.time(), nota=f'Tempo per variabile {v}')
    print()

# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #


if int(config.get('CONCATENAZIONI', 'job')) == 0:
    ### Ciclo sulle variabili
    for v in lista_variabili:
        f_concatenazione(v)

else:
    if config.get('CONCATENAZIONI', 'tipo_di_parallellizzazione') == 'joblib':
        Parallel(n_jobs=int(config.get('CONCATENAZIONI', 'job')), verbose=1000)(delayed(f_concatenazione)(v) for v in lista_variabili)
    
    elif config.get('CONCATENAZIONI', 'tipo_di_parallellizzazione') == 'multiprocessing':
        pool = multiprocessing.Pool(processes=int(config.get('CONCATENAZIONI', 'job')))
        pool.map(f_concatenazione, lista_variabili)
        pool.close()
        pool.join() # Aspetta che tutti finiscano

# %%
# exit(0)

dict_nomi_variabili = {
    'capecon': 'cape',
    'clct': 'tcc',
    'pmsl': 'msl',
    'qv': 'q',
    'r2': 'rh2m',
    'tp': 'tp3',
    'cp': 'cp3',
    }

print('\nCreazione dei dataset delle singole stazioni\n')

lista_stazioni = pd.read_csv(config.get('COMMON', 'percorso_file_coordinate'), index_col=0).index.tolist()
cartella_df_s_tutti_i_punti = f_crea_cartella(f'{cartella_output_concatenazioni}/dataset_tutti_i_punti')
cartella_df_s_A = f_crea_cartella(f'{cartella_output_concatenazioni}/dataset_A')

for s in lista_stazioni:
    f_log_ciclo_for([['Stazione ', s, lista_stazioni]])

    df_s = pd.DataFrame()
    
    for v in lista_variabili:
        df_v = pd.read_csv(f"{cartella_tmp}/df_{v}_{range_previsionale}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv", index_col=0, parse_dates=True)

        try:
            df_v.index = pd.to_datetime(df_v.index, format='ISO8601')
        except ValueError as e:
            logger.error(e)
            raise
            
            ### Questi if non funzionano bene, ho messo format='ISO8601' e sembra gestire bene le date
            if config.get('CONCATENAZIONI', 'modello') == 'MOLOCH' and range_previsionale == '0-24':
                df_v = df_v.drop('2024-01-01', axis=0)
            elif config.get('CONCATENAZIONI', 'modello') == 'ECMWF' and range_previsionale == '24-48':
                df_v = df_v.drop('2022-02-22', axis=0)
            elif config.get('CONCATENAZIONI', 'modello') == 'ECMWF' and range_previsionale == '48-72':
                df_v = df_v.drop('2022-02-23', axis=0)
            
            df_v.index = pd.to_datetime(df_v.index)
            
        df_v_nan = df_v.dropna()
        logger.debug(f'Mancano {df_v.shape[0] - df_v_nan.shape[0]} date al dataset di {v}')
        del (df_v_nan)
        
        ### con gh, quando concateno, i dati mi diventano nan
        # if v == 'gh': stop

        # missing_in_df_s = df_v.index.difference(df_s.index)
        # missing_in_df_v = df_s.index.difference(df_v.index)
        
        df_s = pd.concat([df_s, df_v[[x for x in df_v if s in x]]], axis=1)
        assert not df_s.dropna().shape[0] == 0
        
    del (v)

    ### Rinomino i nomi delle colonne in modo che siano tutte con lo stesso nome tra i diversi modelli
    for chiave, valore in dict_nomi_variabili.items():
        df_s.columns = [x.replace(f'{chiave}_', f'{valore}_') for x in df_s.columns]
        
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'
        
    #####
    ##### La precipitazione di ECMWF ha bisogno di postprocessing per diventare tp3
    ##### ---> non posso farlo qui, devo farlo dentro il primo ciclo -> FATTO
    
    colonne_tp = [x for x in df_s.columns if 'tp' in x or 'cp' in x]
    
    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':
        ### da m a mm
        df_s[[x for x in df_s if x.startswith('tp')]] = df_s[[x for x in df_s if x.startswith('tp')]] * 1000
        df_s[[x for x in df_s if x.startswith('cp')]] = df_s[[x for x in df_s if x.startswith('cp')]] * 1000
        
        df_s[colonne_tp] = df_s[colonne_tp].clip(lower=0) # se è <0, va a 0
    
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'
    
    #####
    ##### Cumulate precipitative
    #####
    
    lista_cumulate = [6, 12, 24] if not dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'MOLOCHsfc' else [3, 6, 12, 24]

    df_s.index = pd.to_datetime(df_s.index)
    df_shift = df_s.copy()

    df_shift.index = df_shift.index - pd.Timedelta(seconds=1)
    lista_df_D = list(df_shift.groupby(pd.Grouper(axis=0, freq='D')))

    dict_prec_cumulate = {}

    for cum in lista_cumulate:
        logger.info(f'cum = {cum}h')
        df_prec_cumulate = pd.DataFrame()
        
        for d0, df0 in lista_df_D:
            
            df_tp_cum = pd.DataFrame()
            for col_tp in colonne_tp:
                
                col_tp_cum = f"{col_tp[0:2]}{cum}_{col_tp.split('_', 1)[-1]}"
                df_tp_cum[col_tp_cum] = df0[col_tp].rolling(f'{cum}h').sum()
            
            df_prec_cumulate = pd.concat([df_prec_cumulate, df_tp_cum], axis=0)

        dict_prec_cumulate[f'{cum}h'] = df_prec_cumulate
        
    for valore in dict_prec_cumulate.values():
        valore.index = valore.index + pd.Timedelta(seconds=1)
        df_s = pd.concat([df_s, valore], axis=1)
    
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'
    
    #####
    ##### Copertura nuvolosa
    #####
    
    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':

        for nuvola in ['tcc', 'hcc', 'mcc', 'lcc']:
            lista_colonne_nuvole = [x for x in df_s.columns if x.startswith(nuvola)]
            df_s[lista_colonne_nuvole] = (df_s[lista_colonne_nuvole] * 100).round(3) # Da (0-1) a %

    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'

    #####
    ##### Modulo e direzione del vento
    #####

    lista_colonne_u = [x for x in df_s.columns if x.startswith('u_') or x.startswith('u2PVU_')]
    lista_colonne_v = [x for x in df_s.columns if x.startswith('v_') or x.startswith('v2PVU_')]

    for col_u, col_v in zip(lista_colonne_u, lista_colonne_v):

        ws = np.sqrt(df_s[col_u] ** 2 + df_s[col_v] ** 2)
        direz = np.deg2rad(wind_direction(df_s[col_u].values * units('m/s'), df_s[col_v].values * units('m/s'), convention='from').magnitude)

        df_ws = pd.DataFrame(ws, columns=[f'ws{col_u[1:]}'], index=df_s.index)
        df_direz = pd.DataFrame(direz, columns=[f'dir{col_u[1:]}'], index=df_s.index)

        df_s = pd.concat([df_s, df_ws, df_direz], axis=1)

    lista_colonne_u10 = [x for x in df_s.columns if x.startswith('u10')]
    lista_colonne_v10 = [x for x in df_s.columns if x.startswith('v10')]
    
    for col_u10, col_v10 in zip(lista_colonne_u10, lista_colonne_v10):
        ws10 = np.sqrt(df_s[col_u10] ** 2 + df_s[col_v10] ** 2)
        direz10 = np.deg2rad(wind_direction(df_s[col_u10].values * units('m/s'), df_s[col_v10].values * units('m/s'), convention='from').magnitude)
    
        df_ws10 = pd.DataFrame(ws10, columns=[col_u10.replace('u10', 'ws10')], index=df_s.index)
        df_direz10 = pd.DataFrame(direz10, columns=[col_u10.replace('u10', 'dir10')], index=df_s.index)
    
        df_s = pd.concat([df_s, df_ws10, df_direz10], axis=1)
    
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'
    
    #####
    ##### Umidità relativa
    #####

    lista_colonne_q = [x for x in df_s.columns if x.startswith('q_')]
    lista_colonne_t = [x for x in df_s.columns if x.startswith('t_')]

    for col_t, col_q in zip(lista_colonne_t, lista_colonne_q):
        livello = int(col_t.split('_')[2])

        df_s[col_q] = df_s[col_q] * 1000 # Da kg/kg a g/kg
        rh = np.array(relative_humidity_from_mixing_ratio(livello * units.hPa, df_s[col_t].values * units.K, df_s[col_q].values * units('g/kg')).to('percent'))
        df_rh = pd.DataFrame(rh, columns=[f'rh{col_t[1:]}'], index=df_s.index)

        df_s = pd.concat([df_s, df_rh], axis=1)

    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':
        lista_colonne_t2m = [x for x in df_s.columns if x.startswith('t2m_')]
        lista_colonne_d2m = [x for x in df_s.columns if x.startswith('d2m_')]

        for col_t2m, col_d2m in zip(lista_colonne_t2m, lista_colonne_d2m):
            rh2m = np.array(relative_humidity_from_dewpoint(df_s[col_t2m].values * units.K, df_s[col_d2m].values * units.K).to('percent'))
            df_rh2m = pd.DataFrame(rh2m, columns=[col_t2m.replace('t2m', 'rh2m')], index=df_s.index)
    
            df_s = pd.concat([df_s, df_rh2m], axis=1)
    
    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] in ['BOLAM', 'MOLOCH']:
        lista_colonne_t2m = [x for x in df_s.columns if x.startswith('t2m_')]
        lista_colonne_rh2m = [x for x in df_s.columns if x.startswith('rh2m_')]
        
        for col_t2m, col_rh2m in zip(lista_colonne_t2m, lista_colonne_rh2m):
            d2m = np.array(dewpoint_from_relative_humidity(df_s[col_t2m].values * units.K, df_s[col_rh2m].values * units.percent))
            df_d2m = pd.DataFrame(d2m, columns=[col_t2m.replace('t2m', 'd2m')], index=df_s.index)
        
            df_s = pd.concat([df_s, df_d2m], axis=1)
    
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'
        
    #####
    ##### Temperatura potenziale
    #####

    for col_t in lista_colonne_t:
        livello = int(col_t.split('_')[2])

        theta = np.array(potential_temperature(livello * units.hPa, df_s[col_t].values * units.K))
        df_theta = pd.DataFrame(theta, columns=[f'theta{col_t[1:]}'], index=df_s.index)
        
        df_s = pd.concat([df_s, df_theta], axis=1)

    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'

    #####
    ##### Temperatura virtuale
    #####

    for col_t, col_q in zip(lista_colonne_t, lista_colonne_q):
        livello = int(col_t.split('_')[2])
            
        T_v = np.array(virtual_temperature(df_s[col_t].values * units.K, df_s[col_q].values * units('g/kg')))
        df_T_v = pd.DataFrame(T_v, columns=[f'Tv{col_t[1:]}'], index=df_s.index)

        df_s = pd.concat([df_s, df_T_v], axis=1)

    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9), f'{df_s.dropna().shape[0]} < {int(df_s.shape[0] * 0.9)}'

    #####
    ##### Seni e coseni di ora e mese
    #####

    if not dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'MOLOCHsfc':
        ora  = df_s.index.hour
        mese = df_s.index.month
        df_cosh = pd.DataFrame(np.cos(2 * np.pi * ora / 23.0), index=df_s.index, columns=['cosh'])
        df_sinh = pd.DataFrame(np.sin(2 * np.pi * ora / 23.0), index=df_s.index, columns=['sinh'])
        df_cosm = pd.DataFrame(np.cos(2 * np.pi * mese / 12.0), index=df_s.index, columns=['cosm'])
        df_sinm = pd.DataFrame(np.sin(2 * np.pi * mese / 12.0), index=df_s.index, columns=['sinm'])
    
        df_s = pd.concat([df_s, df_cosh, df_sinh, df_cosm, df_sinm], axis=1)

    #####
    ##### Dati osservati
    ##### Ho riflettuto ed è meglio avere gli osservati a parte

    #####
    #####
    #####
    
    try:
        df_s = df_s.astype(float).map(f_round, digits=3)
    except AttributeError:
        df_s = df_s.astype(float).applymap(f_round, digits=3)
        
    df_s = df_s.round(4) # va bene per (ad esempio) 0.002976 che deve diventare 0.003
    
    df_s = df_s.sort_index(axis=1)
    df_s = df_s.reindex(pd.date_range(df_s.index[0], df_s.index[-1], freq=f'{(df_s.index[1] - df_s.index[0]).seconds // 3600}h'))
    # df = df.dropna()

    df_nan = df_s[df_s.isna().any(axis=1)]
    logger.warning(f'Mancano {df_nan.shape[0]} date') 
    
    df_feature = pd.read_excel('./feature_modelli.ods', engine='odf', header=0)
    colonne_modello = sorted(df_feature[f"shortName_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}"].dropna().tolist())
    colonne_df_s = sorted(list(set([x.split('_')[0] for x in df_s.columns])))

    for j in colonne_modello:
        if not j in colonne_df_s:
            logger.warning(f'{j} NON in colonne_presenti')
            
    assert colonne_modello == colonne_df_s
    
    assert df_s.dropna().shape[0] > int(df_s.shape[0] * 0.9)

    #####
    ##### Confronto con gli osservati di temperatura
    ##### Funziona, non mi serve più vedere il plot ogni volta
    
    # try:
    #     import matplotlib.pyplot as plt
    #     df_obs = pd.read_csv(f'{cartella_dati_osservati}/temperatura/{s}.csv', index_col=0, parse_dates=True)
    
    #     df_plot = pd.concat([df_obs, df_s[f't2m_A_{s}'] - 273.15], axis=1).dropna().iloc[5000:6000, :]
    #     df_plot.plot()
    #     plt.title(f"{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}, {range_previsionale}")
    #     plt.ylim(-5, 35)
    #     plt.show()
    #     plt.close()

    # except Exception as error:
    #     print(error)
    #     pass

    nome_df_s = f"df_{range_previsionale}_{s}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv"
    df_s.to_csv(f'{cartella_df_s_tutti_i_punti}/{nome_df_s}', index=True, header=True, mode='w', na_rep=np.nan)

    ### Ne approfitto per creare un dataset che contiene solo il punto A
    df_s_A = pd.concat([df_s[[x for x in df_s.columns if '_A_' in x]], df_s[[x for x in df_s.columns if not '_' in x]]], axis=1)
    df_s_A.to_csv(f'{cartella_df_s_A}/{nome_df_s}', index=True, header=True, mode='w', na_rep=np.nan)

# os.system(f'rm -rf {cartella_tmp}')

print('\n\nDone.')
