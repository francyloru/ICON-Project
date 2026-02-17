import argparse
from dati import gestore, unificatore_csv
import gestore_modelli
import cerca_soluzione

import xgboost_train_and_test
import linear_regression_train_and_test
import random_forest_train_and_test

def main():
    parser = argparse.ArgumentParser(description="Script di gestione Dataset e Training")

    # Comandi principali per il progetto
    parser.add_argument("--new_dataset", action="store_true", help="Si considerano nuovi file di meteo presi dalla piattaforma online (https://www.ilmeteo.it/portale/archivio-meteo) e si uniscono (e formalizzano) per essere usati per l'addestramento dei modelli")
    parser.add_argument("--find_models", action="store_true", help="Allena tutte le tipologie di modello di apprendimento su tutte le città, li testa sull'anno 2025 e in base ai risultati dei test, individua il modello migliore per ciascuna città ")
    parser.add_argument("--find_scheduling", action="store_true", help="Esegue l'algoritmo di ricerca A* per trovare la pianificazione che minimizza i costi")

    # Comandi per poter usare i modelli (questi devono essere già allenati)
    parser.add_argument("--use_model_xgboost", action="store_true", help="Lancia il modello xgboost su dei dati di input")
    parser.add_argument("--use_model_random_forest", action="store_true", help="Lancia il modello random forest su dei dati di input")
    parser.add_argument("--use_model_linear_regression", action="store_true", help="Lancia il modello random forest su dei dati di input")

    args = parser.parse_args()

    citta = ['Bari', 'Lecce', 'Potenza']
    path_file = "dati/dataset_meteo_unificato.csv"

    # Esempio di logica: chiama una funzione nell'altro file se il flag è True
    if args.new_dataset:
        print("\n=== LETTURA E FORMALIZZAZIONE DEL DATASET ===")
        unificatore_csv.unifica_dataset(path_file)
        gestore.gestisci_null(path_file)
        gestore.separatore_data(path_file)
        
        gestore.elimina_colonne(path_file, ['PUNTORUGIADA °C', 'VISIBILITA m', 'VENTOMAX km/h', 'RAFFICA km/h', 
        'PRESSIONESLM mb'])
        gestore.aggiungi_ciclicita_data(path_file)
        gestore.aggiungi_temperatura_anno_precedente(path_file)

        #salviamo i dati dell'ultimo anno in un file apposito
        unificatore_csv.dati_ultimo_anno(2025)

    if args.find_models:
        print("\n=== INDIVIDUAZIONE DEL MODELLO MIGLIORE PER OGNI CITTA' ===")
        gestore_modelli.esegui_confronto_e_training(path_file, 'TMEDIA °C', 2025)

    if args.find_scheduling:
        print("\n=== INDIVIDUAZIONE DELLA MIGLIORE PIANIFICAZIONE ===")
        cerca_soluzione.cerca_soluzione()

            
    if args.use_model_xgboost:
        print("")
        ripeti = True
        while ripeti:
            localita = input("Località per cui effettuare la predizione con XGBoost(prima lettera maiuscola):")
            if localita in citta:
                ripeti = False
        xgboost_train_and_test.usa_modello(localita)

    if args.use_model_random_forest:
        print("")
        ripeti = True
        while ripeti:
            localita = input("Località per cui effettuare la predizione con Random Forest(prima lettera maiuscola):")
            if localita in citta:
                ripeti = False
        random_forest_train_and_test.usa_modello(localita)

    if args.use_model_linear_regression:
        print("")
        ripeti = True
        while ripeti:
            localita = input("Località per cui effettuare la predizione con Regressione Lineare(prima lettera maiuscola):")
            if localita in citta:
                ripeti = False
        linear_regression_train_and_test.usa_modello(localita)

if __name__ == "__main__":
    main()