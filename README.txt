
In config.ini, "percorso_file_coordinate" deve essere il path (assoluto) + file del .csv delle coordinate.
Il file deve avere l'indice univoco per ogni stazione, una colonna 'Latitude' e una 'Longitude'.

Ci sono variabili con lo stesso nome (es., u) ma riferite a livelli diversi (es., potentialVorticity o isobaricInhPa)
-> Per risolvere questa omonimia, ho verificato che ogni variabile nell'index ha la tripletta (Index, GRIB_dataType, GRIB_typeOfLevel),
oltre a 'id_ds' dove si trova la variabile. In questo modo posso estrarre le variabili senza che si sovrappongano.

Ci sono alcune variabili (es., cin, cape) che non hanno l'analisi (an). Ovviamente ci sta. Altre variabili (es., u10, v10)
hanno sia l'analisi (an) che la previsione (fc).

 ___ _                              _      _      
| __| |_  _ ______ ___   ___ __ _ _(_)_ __| |_ ___
| _|| | || (_-<_-</ _ \ (_-</ _| '_| | '_ \  _(_-<
|_| |_|\_,_/__/__/\___/ /__/\__|_| |_| .__/\__/__/
                                     |_|          

1) Modifica "config.ini" in modo opportuno.

2) Lancia un run "estrazione_*".

3) Lo script "concatenazioni.py" serve per creare dei grossi dataset per le reti.

4) Lo script "controllo_dataset_finali.py" serve per un ulteriore controllo che la concatenazione abbia funzionato bene.

5) [Opzionale] Lo script "riassunto_stato_cartelle_estrazioni.py" crea un dizionario "dict_camminata.pkl" per avere una
panoramica del contenuto della cartella con tutti i .csv estratti

6) [Opzionale] Lo script "tar_estrazioni.py" serve a creare dei .tar.gz delle cartelle estratte, così da venire essere
più comodo spostarle.

 _____    ___      
|_   _|__|   \ ___ 
  | |/ _ \ |) / _ \
  |_|\___/___/\___/

Fin'ora prendo gli N punti più vicini, vorrei poter prendere anche gli N punti all'interno di un raggio R attorno al punto stazione -> haversine

Per ora la mia funzione di arrotondamento fa il round solo se il numero non è 0.qualcosa;

Aggiungi il salvataggio degli interi campi di precipitazione, vento e temperatura

Fare un salvataggio più intelligente e razionale degli output (pkl ?)
