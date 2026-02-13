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

def dati_ultimo_anno(ultimo_anno):
    file_dati_ultimo_anno = "dati/dati_ultimo_anno/ultimo_anno.csv"
    file_input = "dati/dataset_meteo_unificato.csv"

    with open(file_input, mode="r", newline="", encoding="utf-8") as infile, \
         open(file_dati_ultimo_anno, mode="w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.DictReader(infile, delimiter=";")
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter=";")
        
        # Scrive intestazione
        writer.writeheader()
        
        # Filtra e scrive solo le righe con ANNO == ultimo_anno
        for row in reader:
            if row["ANNO"] == str(ultimo_anno):
                writer.writerow(row)
    
    print(f"  - I dati dell'anno {ultimo_anno} (l'ultimo registrato) sono stati salvati nel file 'ultimo_anno.csv' nella cartella 'dati/dati_ultimo/anno'.")