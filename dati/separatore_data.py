import csv

file_path = "dataset_meteo_unificato.csv"

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

print(f"File '{file_path}' aggiornato correttamente con ANNO, MESE e GIORNO.")
