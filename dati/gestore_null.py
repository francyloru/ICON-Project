import csv

file_path = "dataset_meteo_unificato.csv"

righe_modificate = []

with open(file_path, newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin, delimiter=";")
    fieldnames = reader.fieldnames

    for riga in reader:
        # Se FENOMENI è vuoto o 'NaN', sostituisci con "Soleggiato"
        val = riga.get("FENOMENI", "").strip()
        if val == "" or val.lower() == "":
            riga["FENOMENI"] = "Soleggiato"
        righe_modificate.append(riga)

# Riscrivi il file originale con le modifiche
with open(file_path, "w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(righe_modificate)

print(f"File '{file_path}' aggiornato correttamente.")
