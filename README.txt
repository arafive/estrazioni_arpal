
In config.ini, "percorso_file_coordinate" deve essere il path (assoluto) + file del .csv delle coordinate.
Il file deve avere l'indice univoco per ogni stazione, una colonna 'Latitude' e una 'Longitude'.

Ci sono variabili con lo stesso nome (es., u) ma riferite a livelli diversi (es., potentialVorticity o isobaricInhPa)
-> Per risolvere questa omonimia, ho verificato che ogni variabile nell'index ha la tripletta (Index, GRIB_dataType, GRIB_typeOfLevel),
oltre a 'id_ds' dove si trova la variabile. In questo modo posso estrarre le variabili senza che si sovrappongano.

Ci sono alcune variabili (es., cin, cape) che non hanno l'analisi (an). Ovviamente ci sta. Altre variabili (es., u10, v10)
hanno sia l'analisi (an) che la previsione (fc).

# TODO Fin'ora prendo gli N punti più vicini, vorrei poter prendere anche gli N punti all'interno di un raggio R attorno al punto stazione -> haversine

# TODO Per ora la mia funzione di arrotondamento fa il round solo se il numero non è 0.qualcosa;

# TODO Fare un salvataggio più intelligente e razionale degli output (pkl ?)
