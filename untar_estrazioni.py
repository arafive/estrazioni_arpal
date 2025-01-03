
import os

lista_possibili_cartelle_lavoro = [
    '/media/daniele/Daniele2TB/repo/estrazioni_arpal',
    '/run/media/daniele.carnevale/Daniele2TB/repo/estrazioni_arpal'
]

lista_possibili_cartelle_HD = [
    '/media/daniele',
    '/run/media/daniele.carnevale'
]

cartella_lavoro = [x for x in lista_possibili_cartelle_lavoro if os.path.exists(x)][0]
os.chdir(cartella_lavoro)
del (lista_possibili_cartelle_lavoro)

cartella_HD = [x for x in lista_possibili_cartelle_HD if os.path.exists(x)][0]
del (lista_possibili_cartelle_HD)

from funzioni import f_logger

logger = f_logger('DEBUG')

# %%
cartella_madre_estrazioni = f'{cartella_HD}/Daniele2TB/test/rete_pioggia/Estrazioni_AIxtreme/Liguria'

# modelli = ['BOLAM', 'MOLOCHsfc', 'MOLOCH', 'ECMWF']
modelli = ['ECMWF']

for m in modelli:
    
    if os.path.exists(f'{cartella_madre_estrazioni}/{m}'):
        logger.debug(f'Trovata cartella {cartella_madre_estrazioni}/{m}')
        
        lista_cartella_variabili = sorted(os.listdir(f'{cartella_madre_estrazioni}/{m}/00'))
        
        for v in lista_cartella_variabili:
            lista_forecast = sorted(os.listdir(f'{cartella_madre_estrazioni}/{m}/00/{v}'))
            
            for f in lista_forecast:
                cartella_forecast = f'{cartella_madre_estrazioni}/{m}/00/{v}/{f}'
                lista_file_targz = [x for x in os.listdir(cartella_forecast) if x.endswith('.tar.gz')]
                
                for file_targz in lista_file_targz:
                    nome_cartella_output = file_targz.split('.')[0].split('-')[-1]
                    
                    ### Tolgo questa condizione perché al più la sovrascrivo
                    # if os.path.exists(f'{cartella_forecast}/{nome_cartella_output}'):
                    #     logger.info(f'{cartella_forecast}/{nome_cartella_output} esiste già. Continuo.')
                    #     continue
                    
                    os.chdir(cartella_forecast)
                    comando = f'tar -zxf {cartella_forecast}/{file_targz}'
                    logger.info(comando)
                    os.system(comando)
                    
    else:
        logger.warning(f'Cartella {cartella_madre_estrazioni}/{m} NON trovata. Continuo.')

print('\n\nDone.')
