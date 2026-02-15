import argparse
from dati import gestore, unificatore_csv
import xgboost_train_and_test

def main():
    parser = argparse.ArgumentParser(description="Script di gestione Dataset e Training")

   
    # Definiamo i 4 parametri booleani come flag
    # Se il flag viene messo, il valore diventa True. Se omesso, è False.
    parser.add_argument("--new_dataset", action="store_true", help="Considera un nuovo dataset")

    parser.add_argument("--train_and_test_xgboost", action="store_true", help="Avvia il training e il test con xgboost")
    parser.add_argument("--use_model_xgboost", action="store_true", help="Lancia il modello xgboost su dei dati di input")

    parser.add_argument("--normalize_dataset", action="store_true", help="Normalizza i dati")
    parser.add_argument("--altro_dataset", action="store_true", help="Utilizza il dataset alternativo")

    args = parser.parse_args()

    path_file = "dati/dataset_meteo_unificato.csv"
    localita = 'Lecce'

    # Esempio di logica: chiama una funzione nell'altro file se il flag è True
    if args.new_dataset:
        print("\n--- Lettura e formalizzazione del Dataset:")
        unificatore_csv.unifica_dataset(path_file)
        gestore.gestisci_null(path_file)
        gestore.separatore_data(path_file)
        
        gestore.elimina_colonne(path_file, ['PUNTORUGIADA °C', 'VISIBILITA m', 'VENTOMAX km/h', 'RAFFICA km/h', 
        'PRESSIONESLM mb'])
        gestore.aggiungi_ciclicita_data(path_file)
        gestore.aggiungi_temperatura_anno_precedente(path_file)

        #salviamo i dati dell'ultimo anno in un file apposito
        unificatore_csv.dati_ultimo_anno(2025)
    
    if args.train_and_test_xgboost:
        print("\n--- ADDESTRAMENTO e VALUTAZIONE del Modello con XGBoost:")
        # print(xgboost_train_and_test.predizione_annuale())
        xgboost_train_and_test.train_and_test(path_file, 'TMEDIA °C', localita, 2025)
            
    if args.use_model_xgboost:
        print("\n--- INSERIMENTO dati meteo per PREVISIONE con XGBoost --")
        xgboost_train_and_test.usa_modello(localita)

if __name__ == "__main__":
    main()