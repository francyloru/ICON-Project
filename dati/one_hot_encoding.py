import csv
import os

mesi = [str(i) for i in range(1, 13)]

file_input = "dataset_meteo_unificato.csv"
file_temp = file_input + ".tmp"

with open(file_input, newline="", encoding="utf-8") as fin, \
     open(file_temp, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin, delimiter=";")
    indice_mese = reader.fieldnames.index("MESE")

    fieldnames = (
        reader.fieldnames[:indice_mese] +
        [f"Mese_{m}" for m in mesi] +
        reader.fieldnames[indice_mese+1:]
    )

    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for row in reader:
        mese_corrente = row["MESE"]

        nuova_riga = {k: v for k, v in row.items() if k != "MESE"}

        for m in mesi:
            nuova_riga[f"Mese_{m}"] = "1" if m == mese_corrente else "0"

        writer.writerow(nuova_riga)

# Sostituisco il file originale con quello modificato
os.replace(file_temp, file_input)
