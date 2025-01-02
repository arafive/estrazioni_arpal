
import os

from funzioni import f_logger

logger = f_logger('INFO')

# %%
cartella_madre_estrazioni = '/home/cfmi.arpal.org/daniele.carnevale/Scrivania/estrazioni_arpal/Estrazioni_AIxtreme/Liguria'

# modelli = ['BOLAM', 'MOLOCHsfc', 'MOLOCH', 'ECMWF']
modelli = ['MOLOCHsfc', 'MOLOCH', 'ECMWF']

for m in modelli:
    
    if os.path.exists(f'{cartella_madre_estrazioni}/{m}'):
        logger.debug(f'Trovata cartella {cartella_madre_estrazioni}/{m}')
        
        lista_cartella_variabili = sorted(os.listdir(f'{cartella_madre_estrazioni}/{m}/00'))
        
        for v in lista_cartella_variabili:
            lista_forecast = sorted(os.listdir(f'{cartella_madre_estrazioni}/{m}/00/{v}'))
            
            for f in lista_forecast:
                lista_contenuto_livelli = sorted(os.listdir(f'{cartella_madre_estrazioni}/{m}/00/{v}/{f}'))
                lista_contenuto_livelli_senza_targz = [x for x in lista_contenuto_livelli if not x.endswith('.tar.gz')]
                
                for l in lista_contenuto_livelli_senza_targz:
                    nome_file_targz = f'{m}-00-{v}-{f}-{l}.tar.gz'
                    # print(nome_file_targz)
                    
                    if os.path.exists(f'{cartella_madre_estrazioni}/{m}/00/{v}/{f}/{nome_file_targz}'):
                        logger.info(f'{nome_file_targz} esiste. Continuo.')
                        
                    else:
                        cartella_dove_zippare = f'{cartella_madre_estrazioni}/{m}/00/{v}/{f}'
                        os.chdir(cartella_dove_zippare)
                        
                        comando = f'tar -zcf ./{nome_file_targz} {l}'
                        logger.info(comando)
                        os.system(comando)
    else:
        logger.warning(f'Cartella {cartella_madre_estrazioni}/{m} NON trovata. Continuo.')

print('\n\nDone.')

# rsync -arzhPv --prune-empty-dirs --size-only --info=progress2 --include="*/" --include="*.tar.gz" --exclude="*" daniele.carnevale@01588-lenovo.cfmi.arpal.org:/home/cfmi.arpal.org/daniele.carnevale/Scrivania/estrazioni_arpal/Estrazioni_AIxtreme/Liguria/ECMWF /media/daniele/Daniele2TB/test/rete_pioggia/Estrazioni_AIxtreme/Liguria/.

### Note per il trasferimento con rsync dei soli file *.tar.gz
# rsync -av --prune-empty-dirs --include="*/" --include="*.tar.gz" --exclude="*" /source/directory/ /destination/directory/

# Explanation:
# -a: Archive mode (preserves permissions, timestamps, symbolic links, etc.).
# -v: Verbose (shows what is being transferred).
# --prune-empty-dirs: Ensures that directories without .tar.gz files are not copied, even if they exist in the source.
# --include="*/": Ensures that the directory structure is copied, even if the directories don't contain .tar.gz files.
# --include="*.tar.gz": Includes only files ending with .tar.gz.
# --exclude="*": Excludes everything else that doesn't match the previous include patterns.
# /source/directory/: The source directory you want to copy from.
# /destination/directory/: The destination directory you want to copy to.
