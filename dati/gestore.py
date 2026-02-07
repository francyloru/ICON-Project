import csv
import os

# per gestire i valori NULL e i valori numerici con la virgola(al posto del punto)
def gestisci_null(file_path):
    #file_path = "dati/dataset_meteo_unificato.csv"

    # Lista delle colonne che contengono numeri con la virgola da convertire
    colonne_numeriche = [
        'TMEDIA °C', 'TMIN °C', 'TMAX °C', 'PUNTORUGIADA °C', 
        'VENTOMEDIA km/h', 'VENTOMAX km/h', 'RAFFICA km/h', 
        'PRESSIONESLM mb', 'PRESSIONEMEDIA mb', 'PIOGGIA mm'
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

# rappresenta i mesi e le città con la tecnica dell'one_hot_encoding
def one_hot_encoding(file_input, colonne_da_trasformare):
    # colonne da trasformare = ["MESE", "LOCALITA"]
    file_temp = file_input + ".tmp"

    with open(file_input, newline="", encoding="utf-8") as fin:
        # Leggiamo prima tutto per identificare le categorie uniche delle città
        reader = csv.DictReader(fin, delimiter=";")
        righe = list(reader)
        nomi_colonne_originali = reader.fieldnames

    # Determiniamo le categorie per ogni colonna
    mappa_categorie = {}
    for col in colonne_da_trasformare:
        if col == "MESE":
            # Per i mesi forziamo 1-12 anche se nel dataset ne mancasse uno
            mappa_categorie[col] = [str(i) for i in range(1, 13)]
        else:
            # Per le città (o altro) prendiamo i valori unici presenti nel file
            categorie_uniche = sorted(list(set(riga[col] for riga in righe if riga[col])))
            mappa_categorie[col] = categorie_uniche

    # Costruiamo i nuovi fieldnames
    nuovi_fieldnames = []
    for col in nomi_colonne_originali:
        if col in colonne_da_trasformare:
            # Sostituiamo la colonna con le sue versioni One-Hot
            prefix = "Mese" if col == "MESE" else col
            for cat in mappa_categorie[col]:
                nuovi_fieldnames.append(f"{prefix}_{cat}")
        else:
            nuovi_fieldnames.append(col)

    # Scriviamo il file modificato
    with open(file_temp, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=nuovi_fieldnames, delimiter=";")
        writer.writeheader()

        for riga in righe:
            nuova_riga = {}
            # Copiamo le colonne che non cambiano
            for col in nomi_colonne_originali:
                if col not in colonne_da_trasformare:
                    nuova_riga[col] = riga[col]
            
            # Espandiamo le colonne One-Hot
            for col in colonne_da_trasformare:
                valore_corrente = riga[col]
                prefix = "Mese" if col == "MESE" else col
                for cat in mappa_categorie[col]:
                    nuova_riga[f"{prefix}_{cat}"] = "1" if cat == valore_corrente else "0"
            
            writer.writerow(nuova_riga)

    os.replace(file_temp, file_input)
    print(f"  - La rappresentazione One-Hot Encoding è stata applicat a: {', '.join(colonne_da_trasformare)}.\n")
