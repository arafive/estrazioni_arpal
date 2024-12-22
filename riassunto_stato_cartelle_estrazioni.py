
import os
import pickle

lista_possibili_cartelle_lavoro = [
    '/media/daniele/Daniele2TB/repo/estrazioni_arpal',
    '/run/media/daniele.carnevale/Daniele2TB/repo/estrazioni_arpal',
]

lista_possibili_cartelle_HD = [
    '/media/daniele',
    '/run/media/daniele.carnevale',
]

cartella_lavoro = [x for x in lista_possibili_cartelle_lavoro if os.path.exists(x)][0]
os.chdir(cartella_lavoro)
del (lista_possibili_cartelle_lavoro)

cartella_HD = [x for x in lista_possibili_cartelle_HD if os.path.exists(x)][0]
del (lista_possibili_cartelle_HD)

from funzioni import f_logger

logger = f_logger('INFO')

# %%

cartella_madre_estrazioni = f'{cartella_HD}/Caradhras/Estrazioni_AIxtreme/Liguria/nuova_estrazione'

dict_camminata = {'': {}}

### https://stackoverflow.com/questions/1878247/python-directory-searching-and-organizing-by-dict
for cartella, dirs_dentro, files_dentro in os.walk(cartella_madre_estrazioni):
    d = dict_camminata
    cartella = cartella[len(cartella_madre_estrazioni):]
    logger.info(cartella)
    
    for subd in cartella.split(os.sep):
        based = d
        d = d[subd]
        
    if dirs_dentro:
        for dn in dirs_dentro:
            d[dn] = {}
    else:
        based[subd] = files_dentro

pickle.dump(dict_camminata, open(f'{cartella_lavoro}/dict_camminata.pkl', 'wb'))

print('\n\nDone.')
