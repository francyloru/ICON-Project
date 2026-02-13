import csv
import os
import math
from datetime import datetime

# per gestire i valori NULL e i valori numerici con la virgola(al posto del punto)
def gestisci_null(file_path):
    #file_path = "dati/dataset_meteo_unificato.csv"

    # Lista delle colonne che contengono numeri con la virgola da convertire
    colonne_numeriche = [
        'TMEDIA °C', 'TMIN °C', 'TMAX °C', 
        'VENTOMEDIA km/h', 'PRESSIONEMEDIA mb', 'PIOGGIA mm'
    ]

    righe_modificate = []

    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        fieldnames = reader.fieldnames

        for riga in reader:
            # 1. GESTIONE NULL SU FENOMENI
            val = riga.get("FENOMENI", "").strip()
            if val == "" or val.lower() == "":
                riga["FENOMENI"] = "soleggiato"

            # 2. GESTIONE VIRGOLE -> PUNTI
            for col in colonne_numeriche:
                if col in riga and riga[col]:
                    # Sostituisce la virgola con il punto
                    riga[col] = riga[col].replace(",", ".")
            
            righe_modificate.append(riga)

    # Riscrivi il file originale con le modifiche
    with open(file_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(righe_modificate)

    print("  - Sono stati gestiti i NULL.\n  - Sono stati convertiti i valori decimali nel formato puntato.")

# suddivide il campo DATA in ANNO, MESE e Giorno
def separatore_data(file_path):

    # file_path = "dati/dataset_meteo_unificato.csv"

    righe_modificate = []

    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        fieldnames = reader.fieldnames.copy()

        # Rimuovi la colonna 'DATA' e inserisci 'ANNO', 'MESE', 'GIORNO' al suo posto
        if "DATA" in fieldnames:
            idx = fieldnames.index("DATA")
            # rimuovo DATA
            fieldnames.pop(idx)
            # inserisco ANNO, MESE, GIORNO nella stessa posizione
            fieldnames[idx:idx] = ["ANNO", "MESE", "GIORNO"]

        for riga in reader:
            data_str = riga.get("DATA", "")
            # Default valori
            anno, mese, giorno = "", "", ""

            if data_str:
                # con formato data GIORNO/MESE/ANNO
                parti = data_str.split("/")
                if len(parti) == 3:
                    giorno, mese, anno = parti

            # Rimuovo la vecchia colonna DATA
            if "DATA" in riga:
                del riga["DATA"]

            # Aggiungo le nuove colonne
            riga["ANNO"] = anno
            riga["MESE"] = mese
            riga["GIORNO"] = giorno

            righe_modificate.append(riga)

    # Riscrivi il file originale con le modifiche
    with open(file_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(righe_modificate)

    print("  - Il campo DATA è stato diviso in ANNO, MESE e GIORNO.")

def elimina_colonne(file_path, colonne_da_eliminare):
    righe_modificate = []
    
    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        
        # Copia i fieldnames attuali per modificarli
        fieldnames = reader.fieldnames.copy()
        
        # Rimuovi le colonne dall'intestazione (se esistono)
        for col in colonne_da_eliminare:
            if col in fieldnames:
                fieldnames.remove(col)
        
        # Itera sulle righe e rimuovi i dati
        for riga in reader:
            for col in colonne_da_eliminare:
                # Rimuovi la chiave dal dizionario se presente
                if col in riga:
                    del riga[col]
            righe_modificate.append(riga)

    # Riscrivi il file senza le colonne rimosse
    with open(file_path, "w", newline="", encoding="utf-8") as fout:
        # 'extrasaction' non è strettamente necessario qui perché abbiamo rimosso 
        # le chiavi manualmente dal dizionario 'riga', ma è una buona sicurezza.
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(righe_modificate)

    print(f"  - Sono state eliminate le Colonne: {colonne_da_eliminare}")


def aggiungi_ciclicita_data(file_path):
   
    righe_modificate = []

    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        fieldnames = reader.fieldnames.copy()

        # Inserisci le nuove colonne nell'header subito dopo 'GIORNO'
        if "GIORNO" in fieldnames:
            idx = fieldnames.index("GIORNO")
            # Inseriamo dopo GIORNO (idx + 1)
            fieldnames.insert(idx + 1, "COS_GIORNO")
            fieldnames.insert(idx + 1, "SIN_GIORNO") # Inseriamo prima SIN così finisce prima di COS
        else:
            # Fallback se non trova GIORNO: le aggiunge alla fine
            fieldnames.extend(["SIN_GIORNO", "COS_GIORNO"])

        for riga in reader:
            try:
                # Recupera anno, mese, giorno convertendoli in interi
                anno = int(riga.get("ANNO", 0))
                mese = int(riga.get("MESE", 0))
                giorno = int(riga.get("GIORNO", 0))
                
                # Crea un oggetto data per ottenere il numero del giorno nell'anno (1-366)
                data_obj = datetime(anno, mese, giorno)
                giorno_anno = data_obj.timetuple().tm_yday
                
                # Giorni in un anno medio (considerando i bisestili per l'apprendimento)
                giorni_totali = 365.25
                
                # Calcolo Sin e Cos
                # Formula: sin( 2 * pi * giorno_corrente / giorni_totali )
                val_sin = math.sin(2 * math.pi * giorno_anno / giorni_totali)
                val_cos = math.cos(2 * math.pi * giorno_anno / giorni_totali)
                
                # Aggiungi i valori alla riga (arrotondati a 5 decimali)
                riga["SIN_GIORNO"] = round(val_sin, 5)
                riga["COS_GIORNO"] = round(val_cos, 5)

            except (ValueError, TypeError):
                # Se la data non è valida o mancano i dati, metti 0 o lascia vuoto
                riga["SIN_GIORNO"] = 0
                riga["COS_GIORNO"] = 0

            righe_modificate.append(riga)

    # Scrittura su file
    with open(file_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(righe_modificate)

    print("  - Aggiunte le Colonne SIN_GIORNO e COS_GIORNO per la ciclicità temporale.")

def aggiungi_temperatura_anno_precedente(file_path):
    righe_modificate = []
    # Dizionario per mappare (anno, mese, giorno) -> temperatura
    mappa_temperature = {}

    # 1. Lettura preliminare per caricare i dati in memoria
    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        fieldnames = reader.fieldnames.copy()
        
        for riga in reader:
            try:
                chiave = (int(riga['ANNO']), int(riga['MESE']), int(riga['GIORNO']))
                mappa_temperature[chiave] = riga.get('TMEDIA °C', "")
            except (ValueError, KeyError):
                continue

    # 2. Aggiunta della nuova colonna nell'header
    nuova_colonna = 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE'
    if nuova_colonna not in fieldnames:
        # La inseriamo magari dopo TMEDIA °C se esiste, o in fondo
        if 'TMEDIA °C' in fieldnames:
            idx = fieldnames.index('TMEDIA °C')
            fieldnames.insert(idx + 1, nuova_colonna)
        else:
            fieldnames.append(nuova_colonna)

    # 3. Seconda passata per calcolare il valore dell'anno precedente
    with open(file_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        for riga in reader:
            # se non abbiamo la temperatura dell'anno precedente, mettiamo la media dell'anno stesso
            temp = riga['TMEDIA °C']
            try:
                anno_prec = int(riga['ANNO']) - 1
                mese = int(riga['MESE'])
                giorno = int(riga['GIORNO'])
                
                # Cerchiamo nella mappa se esiste il valore per l'anno precedente
                valore_prec = mappa_temperature.get((anno_prec, mese, giorno), "")
                if not valore_prec:
                    
                    riga[nuova_colonna] = riga['TMEDIA °C']
                    
                else:
                    riga[nuova_colonna] = valore_prec
            except (ValueError, KeyError):
                riga[nuova_colonna] = temp
            
            righe_modificate.append(riga)

    # 4. Scrittura finale
    with open(file_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(righe_modificate)

    print(f"  - Aggiunta la Colonna '{nuova_colonna}'.")


# legge la temperatura media del giorno indicato nell'ultimo anno
def leggi_tmedia(mese, giorno):
    file_input = "dati/dati_ultimo_anno/ultimo_anno.csv"

    with open(file_input, mode="r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile, delimiter=';')

        # Normalizza intestazioni
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            # Normalizza chiavi
            row = {k.strip(): v.strip() for k, v in row.items()}

            if (row.get("MESE") == str(mese) and
                row.get("GIORNO") == str(giorno)):
                
                return row.get("TMEDIA °C")

    return None  # se non trova la riga