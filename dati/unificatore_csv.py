import csv
import os

cartella_input = "dati/dati_meteo_separati_csv"
prima_volta = False

mesi = {
    "Gennaio": 1, "Febbraio": 2, "Marzo": 3, "Aprile": 4,
    "Maggio": 5, "Giugno": 6, "Luglio": 7, "Agosto": 8,
    "Settembre": 9, "Ottobre": 10, "Novembre": 11, "Dicembre": 12
}

def chiave_ordinamento(nome_file):
    nome = os.path.splitext(nome_file)[0]
    parti = nome.split("-")

    if len(parti) < 3:
        return (0, 0)

    anno = parti[-2]
    mese = parti[-1]

    return int(anno), mesi.get(mese, 0)

# per creare il dataset unico
def unifica_dataset(file_output):
    prima_volta = True
    
    with open(file_output, "w", newline="", encoding="utf-8") as fout:
        writer = None

        for nome_file in sorted(os.listdir(cartella_input), key=chiave_ordinamento):
            if nome_file.lower().endswith(".csv"):
                with open(os.path.join(cartella_input, nome_file),
                        newline="", encoding="utf-8") as fin:

                    reader = csv.DictReader(fin, delimiter=";")

                    if prima_volta:
                        writer = csv.DictWriter(
                            fout,
                            fieldnames=reader.fieldnames,
                            delimiter=";"
                        )
                        writer.writeheader()
                        prima_volta = False

                    for riga in reader:
                        writer.writerow(riga)

    print("  - E' stato creato il dataset unificato.")

