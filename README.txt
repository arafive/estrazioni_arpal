
In config.ini, "percorso_file_coordinate" deve essere il path (assoluto) + file del .csv delle coordinate.
Il file deve avere l'indice univoco per ogni stazione, una colonna 'Latitude' e una 'Longitude'.

Ci sono variabili con lo stesso nome (es, u) ma riferite a livelli diversi (es, potentialVorticity o isobaricInhPa)
-> Per risolvere questa omonimia, ho verificato che ogni variabile nell'index ha la tripletta (Index, GRIB_dataType, GRIB_typeOfLevel),
oltre a 'id_ds' dove si trova la variabile. In questo modo posso estrarre le variabili senza che si sovrappongano.
