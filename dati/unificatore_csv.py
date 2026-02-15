import csv
import os
from datetime import datetime

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

def unifica_dataset(file_output):
    tutte_righe = []
    fieldnames = None

    # Legge tutti i file
    for nome_file in os.listdir(cartella_input):
        if nome_file.lower().endswith(".csv"):
            percorso_file = os.path.join(cartella_input, nome_file)

            with open(percorso_file, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin, delimiter=";")

                # Normalizza intestazioni
                current_fieldnames = [f.strip() for f in reader.fieldnames]

                # 🔥 Se esiste VISIBILITA km, rinominala in VISIBILITA m
                if "VISIBILITA km" in current_fieldnames:
                    idx = current_fieldnames.index("VISIBILITA km")
                    current_fieldnames[idx] = "VISIBILITA m"

                # Salviamo le intestazioni una sola volta
                if fieldnames is None:
                    fieldnames = current_fieldnames

                for riga in reader:
                    riga = {k.strip(): v.strip() for k, v in riga.items()}

                    # 🔥 Rinomina anche nei dati
                    if "VISIBILITA km" in riga:
                        riga["VISIBILITA m"] = riga.pop("VISIBILITA km")

                    tutte_righe.append(riga)

                    
    # Ordina prima per LOCALITA, poi per DATA
    def chiave_ordinamento(riga):
        localita = riga.get("LOCALITA", "")
        data_str = riga.get("DATA", "01/01/1900")
        # Converte gg/mm/aaaa in datetime per ordinamento corretto
        try:
            data = datetime.strptime(data_str, "%d/%m/%Y")
        except ValueError:
            data = datetime(1900, 1, 1)  # default in caso di errore
        return (localita.lower(), data)

    tutte_righe.sort(key=chiave_ordinamento)

    # Scrive file unificato
    with open(file_output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(tutte_righe)

    print(f"  - E' stato creato il dataset unificato ordinato per città e data in '{file_output}'.")



def dati_ultimo_anno(ultimo_anno):
    cartella_output = "dati/dati_ultimo_anno"
    file_input = "dati/dataset_meteo_unificato.csv"

    os.makedirs(cartella_output, exist_ok=True)

    # Dizionario: {localita: lista_righe}
    dati_per_citta = {}

    with open(file_input, mode="r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile, delimiter=";")

        # Normalizza nomi colonne (evita problemi di spazi)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}

            if row.get("ANNO") == str(ultimo_anno):
                localita = row.get("LOCALITA")

                if localita not in dati_per_citta:
                    dati_per_citta[localita] = []

                dati_per_citta[localita].append(row)

    # Creazione file per ogni città
    for localita, righe in dati_per_citta.items():
        nome_file = f"ultimo_anno_{localita}.csv"
        percorso_file = os.path.join(cartella_output, nome_file)

        with open(percorso_file, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(righe)

        print(f"           - Creato file contenente i dati dell'anno {ultimo_anno} a {localita}: {nome_file}")

    print(f"  - Tutti i file dell'anno {ultimo_anno} sono stati salvati in '{cartella_output}'.")

 