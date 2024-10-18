
import os
import time
import configparser

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

from funzioni import f_log_ciclo_for
from funzioni import f_crea_cartella
from funzioni import f_printa_tempo_trascorso


config = configparser.ConfigParser()
config.read('./config.ini')

dict_config_modelli = {
    'ECITA': 'ECMWF',
    'BOLAM': 'BOLAM',
    'MOLOCH': 'MOLOCH',
    'MOLOCHsfc': 'MOLOCHsfc',
}

cartella_madre_estrazione = f"{config.get('COMMON', 'cartella_madre_estrazione')}/{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}"
ora_start_forecast = f"{config.get('COMMON', 'ora_start_forecast')}"

cartella_dati_osservati = f"{config.get('CONCATENAZIONI', 'cartella_dati_osservati')}"
cartella_madre_output_concatenazioni = f"{config.get('CONCATENAZIONI', 'cartella_madre_output_concatenazioni')}"
range_previsionale = f"{config.get('CONCATENAZIONI', 'range_previsionale')}"

cartella_output_concatenazioni = f_crea_cartella(f"{cartella_madre_output_concatenazioni}/{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}/{ora_start_forecast}/{range_previsionale}")
cartella_tmp = f_crea_cartella(f'{cartella_output_concatenazioni}/tmp', print_messaggio=False)

# %%

lista_variabili = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}'))

def f_concatenazione(v):
# for v in lista_variabili:
    print(v)
    
    df_v = pd.DataFrame()

    # lista_cartelle_an_fc = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}'))
    # for f in lista_cartelle_an_fc:
    f = 'fc'  # Non prendo l'analisi

    lista_livelli = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}'))

    for l in lista_livelli:
        if l == 'potentialVorticity':
            v_nome = f'{v}2PVU'
        else:
            v_nome = v

        global lista_stazioni
        lista_stazioni = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}'))

        for s in lista_stazioni:
        # for s in lista_stazioni[0:3]:
            t_inizio_s = time.time()
            f_log_ciclo_for([['Variabile ', v, lista_variabili],
                             ['Stazione ', s, lista_stazioni]])

            df_s = pd.DataFrame()

            lista_file_tempi = sorted(os.listdir(f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}/{s}'))
            # lista_file_tempi = lista_file_tempi[0:5]
            lista_datetime = pd.to_datetime([x.split('.')[0] for x in lista_file_tempi])

            for t, d in zip(lista_file_tempi, lista_datetime):
                # print(v, f, l, s, str(t))

                cartella_df = f'{cartella_madre_estrazione}/{ora_start_forecast}/{v}/{f}/{l}/{s}'

                try:
                    df = pd.read_csv(f'{cartella_df}/{t}', index_col=0, parse_dates=True)
                except pd.errors.EmptyDataError:
                    ### ECMWF/00/cape/fc/surface/TESTI/2021-02-03.csv erano vuoti, non so perchè
                    continue

                if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF' and v in ['tp', 'cp']:
                    ### Ecita: non ha la tp3 ma una precipitazione cumualta dallo start fino alla fine
                    df_shift = df.shift(periods=1, fill_value=0) # periods=1 per la tp3, nota bene che oltre +96 diventa ogni 6 ore
                    df = df - df_shift

                if df.index[0].hour == 0:
                    ### Ecita: dal 2023-01-14 l'analisi e il forecast non sono più separati. Devo togliere la prima riga
                    df = df.drop(df.index[0], axis=0)

                if '.' in df.columns[0]:
                    ### Colpa mia, solo nell'ecita
                    df.columns = [x.split('.')[0] for x in df.columns]
                    df.columns = [f"{x.split('_')[1]}_{x.split('_')[0]}" for x in df.columns]

                freq = '1h' if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'MOLOCHsfc' else '3h'
                df = df.loc[df.index.intersection(pd.date_range(d + pd.DateOffset(hours=int(range_previsionale.split('-')[0])),
                                                                d + pd.DateOffset(hours=int(range_previsionale.split('-')[1])),
                                                                freq=freq))]

                v_nome = v_nome.replace('_', '')  # Nel caso di 'cape_con' -> 'capecon'
                df.columns = [f'{v_nome}_{x}_{s}' for x in df.columns]

                df_s = pd.concat([df_s, df], axis=0)

            try:
                df_v = pd.concat([df_v, df_s], axis=1)
            except pd.errors.InvalidIndexError:
                ### Problemi con ECMWF/00/cape/fc/surface/TESTI/2023-12-31.csv
                continue

    lista_completa_datetime = pd.date_range(lista_datetime[0], lista_datetime[-1] + pd.DateOffset(hours=24), freq=freq)

    ### Per non rischiare di generare un mostro di .csv, devo salvare a pezzetti.

    if range_previsionale == '24-48' or range_previsionale == '48-72':
        ### Tengo la prima previsione successiva
        df_v = df_v[~df_v.index.duplicated(keep='last')]
        
    df_v = df_v.reindex(lista_completa_datetime, fill_value=np.nan)
    df_v = df_v.sort_index()
    df_v = df_v.dropna()

    df_v.to_csv(f"{cartella_tmp}/df_{v}_{range_previsionale}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv", index=True, header=True, mode='w', na_rep=np.nan)

    f_printa_tempo_trascorso(t_inizio_s, time.time(), nota=f'Tempo per variabile {v}')
    print()

# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #
# # # # # # # #   # # # # # # # #   # # # # # # # #


if int(config.get('CONCATENAZIONI', 'job_joblib')) == 0:
    ### Ciclo sulle variabili
    for v in lista_variabili:
        f_concatenazione(v)
    
else:
    Parallel(n_jobs=int(config.get('CONCATENAZIONI', 'job_joblib')), verbose=1000)(delayed(f_concatenazione)(v) for v in lista_variabili)
    
# %%

print('\nCreazione dei dataset delle singole stazioni\n')

for s in lista_stazioni:
# for s in lista_stazioni[0:3]:
    f_log_ciclo_for([['Stazione ', s, lista_stazioni]])

    df_s = pd.DataFrame()

    for v in lista_variabili:
        df_v = pd.read_csv(f"{cartella_tmp}/df_{v}_{range_previsionale}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv", index_col=0, parse_dates=True)
        df_s = pd.concat([df_s, df_v[[x for x in df_v if s in x]]], axis=1)
    del (v)

    #####
    ##### La precipitazione di ECMWF ha bisogno di postprocessing per diventare tp3
    ##### ---> non posso farlo qui, devo farlo dentro il primo ciclo -> FATTO

    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':
        ### da m a mm
        df_s[[x for x in df_s if x.startswith('tp_')]] = df_s[[x for x in df_s if x.startswith('tp_')]] * 1000
        df_s[[x for x in df_s if x.startswith('cp_')]] = df_s[[x for x in df_s if x.startswith('cp_')]] * 1000

    #####
    ##### Modulo e direzione del vento
    #####

    lista_colonne_u = [x for x in df_s.columns if x.startswith('u_') or x.startswith('u2PVU_')]

    for col_u in lista_colonne_u:
        col_v = f'v{col_u[1:]}'

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
    
    #####
    ##### Umidità relativa
    #####

    lista_colonne_q = [x for x in df_s.columns if x.startswith('q_') or x.startswith('qv_')]
    lista_colonne_t = [x for x in df_s.columns if x.startswith('t_')]

    for col_t, col_q in zip(lista_colonne_t, lista_colonne_q):
        livello = int(col_t.split('_')[1])

        df_s[col_q] = df_s[col_q] * 1000 # Da kg/kg a g/kg
        rh = np.array(relative_humidity_from_mixing_ratio(livello * units.hPa, df_s[col_t].values * units.K, df_s[col_q].values * units('g/kg')).to('percent'))
        df_rh = pd.DataFrame(rh, columns=[f'rh{col_t[1:]}'], index=df_s.index)

        df_s = pd.concat([df_s, df_rh], axis=1)
        
    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] == 'ECMWF':
        lista_colonne_t2m = [x for x in df_s.columns if x.startswith('t2m_')]
        lista_colonne_d2m = [x for x in df_s.columns if x.startswith('d2m_')]

        for col_t2m, col_d2m in zip(lista_colonne_t2m, lista_colonne_d2m):
            rh2m = np.array(relative_humidity_from_dewpoint(df_s[col_t2m].values * units.K, df_s[col_d2m].values).to('percent'))
            df_rh2m = pd.DataFrame(rh2m, columns=[col_t2m.replace('t2m', 'rh2m')], index=df_s.index)
    
            df_s = pd.concat([df_s, df_rh2m], axis=1)
            
    if dict_config_modelli[config.get('CONCATENAZIONI', 'modello')] in ['BOLAM', 'MOLOCH']:
        lista_colonne_t2m = [x for x in df_s.columns if x.startswith('t2m_')]
        lista_colonne_rh2m = [x for x in df_s.columns if x.startswith('rh2m_')]
        
        for col_t2m, col_rh2m in zip(lista_colonne_t2m, lista_colonne_rh2m):
            d2m = np.array(dewpoint_from_relative_humidity(df_s[col_t2m].values * units.K, df_s[col_rh2m].values * units.percent))
            df_d2m = pd.DataFrame(d2m, columns=[col_t2m.replace('t2m', 'd2m')], index=df_s.index)
        
            df_s = pd.concat([df_s, df_rh2m], axis=1)
        
    #####
    ##### Temperatura potenziale
    #####

    for col_t in lista_colonne_t:
        livello = int(col_t.split('_')[1])
        theta = np.array(potential_temperature(livello * units.hPa, df_s[col_t].values * units.K))
        df_theta = pd.DataFrame(theta, columns=[f'theta{col_t[1:]}'], index=df_s.index)
        
        df_theta = pd.concat([df_s, df_theta], axis=1)

    #####
    ##### Temperatura virtuale
    #####

    for col_t, col_q in zip(lista_colonne_t, lista_colonne_q):
        livello = int(col_t.split('_')[1])

        T_v = np.array(virtual_temperature(df_s[col_t].values * units.K, df_s[col_q].values * units('g/kg')))
        df_T_v = pd.DataFrame(T_v, columns=[f'Tv{col_t[1:]}'], index=df_s.index)

        df_s = pd.concat([df_s, df_T_v], axis=1)

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
    #####

    # # TODO fai dei test -> hi riflettuto ed è meglio averli a parte
    # for cartella in ['direzione', 'modulo', 'precipitazione', 'temperatura']:

    #     if not cartella == 'precipitazione':

    #         if os.path.exists(f'{cartella_dati_osservati}/{cartella}/{s}.csv'):
    #             df_obs = pd.read_csv(f'{cartella_dati_osservati}/{cartella}/{s}.csv', index_col=0, parse_dates=True)
    #             df_s = pd.concat([df_s, df_obs], axis=1)
            
    #     else:
    #         for i in ['1H', '3H', '6H', '12H', '24H']:

    #             if os.path.exists(f'{cartella_dati_osservati}/{cartella}/{i}/{s}.csv'):
    #                 df_obs = pd.read_csv(f'{cartella_dati_osservati}/{cartella}/{i}/{s}.csv', index_col=0, parse_dates=True)
    #                 df_s = pd.concat([df_s, df_obs], axis=1)

    df_s = df_s.dropna()

    df_s.to_csv(f"{cartella_output_concatenazioni}/df_{range_previsionale}_{s}_{dict_config_modelli[config.get('CONCATENAZIONI', 'modello')]}_{config.get('CONCATENAZIONI', 'regione')}.csv", index=True, header=True, mode='w', na_rep=np.nan)

# os.system(f'rm -rf {cartella_tmp}')

print('\n\nDone.')
