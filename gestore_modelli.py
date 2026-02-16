import pandas as pd
import json
import os
from dati.gestore import leggi_tmedia

# Importiamo i moduli dei modelli
import xgboost_train_and_test
import random_forest_train_and_test
import linear_regression_train_and_test
# import extratrees_train_and_test

# Mappa per richiamare i moduli dinamicamente
MAPPA_MODELLI = {
    'xgboost': xgboost_train_and_test,
    'random_forest': random_forest_train_and_test,
    'linear_regression': linear_regression_train_and_test,
    # 'extratrees': extratrees_train_and_test
}

FILE_CONFIG_BEST_MODELS = 'modelli/migliori_modelli.json'

def esegui_confronto_e_training(dataset, target_column, anno_test):
    citta_list = ['Bari', 'Lecce', 'Potenza']
    modelli_nomi = list(MAPPA_MODELLI.keys())
    
    # Dizionario per accumulare i risultati: { 'modello': {'Bari': rmse, 'Lecce': rmse...} }
    risultati_matrix = {m: {} for m in modelli_nomi}
    
    migliori_modelli = {} # { 'Bari': 'xgboost', ... }

    print("=== INIZIO TRAINING E CONFRONTO MODELLI ===")

    for localita in citta_list:
        print(f"\n>>>> Elaborazione città: {localita}")
        best_rmse = float('inf')
        best_model_name = None

        for nome_modello, modulo in MAPPA_MODELLI.items():
            print(f"   > Training modello: {nome_modello}")
            
            # Eseguiamo il train_and_test del modulo specifico
            rmse = modulo.train_and_test(dataset, target_column, localita, anno_test)
            
            # Salviamo il risultato per la matrice
            risultati_matrix[nome_modello][localita] = rmse
            
            # Controlliamo se è il migliore per questa città
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = nome_modello

            print("")        
        migliori_modelli[localita] = best_model_name
        print(f"   *** Miglior modello per {localita}: {best_model_name} (RMSE: {best_rmse}) ***")

    # Creazione Matrice di Confronto (DataFrame)
    df_confronto = pd.DataFrame(risultati_matrix)
    # Trasponiamo per avere Modelli sulle righe e Città sulle colonne (come richiesto)
    df_confronto = df_confronto.T 
    
    # Visualizzazione
    print("\n" + "="*50)
    print(f"   MATRICE DI CONFRONTO RMSE (per il {anno_test})")
    print("="*50)
    print(df_confronto)
    print("="*50)
    
    # Evidenziare il minimo per ogni città (colonna)
    print("\nRIEPILOGO MIGLIORI MODELLI:")
    for citta in citta_list:
        modello_top = migliori_modelli[citta]
        valore_rmse = risultati_matrix[modello_top][citta]
        print(f" - {citta}: {modello_top} (RMSE: {valore_rmse})")

    # Salvataggio associazione Città-Modello su JSON
    os.makedirs('dati', exist_ok=True)
    with open(FILE_CONFIG_BEST_MODELS, 'w') as f:
        json.dump(migliori_modelli, f, indent=4)
    
    print(f"\nAssociazione modelli migliori salvata in '{FILE_CONFIG_BEST_MODELS}'")

    # Salvataggio matrice su CSV opzionale
    df_confronto.to_csv('dati/confronto_rmse_modelli.csv', sep=';', decimal=',')


def predici_temperatura_localita(localita, anno, mese, giorno):
    """
    Individua il modello migliore per la località, recupera la t_media anno prec
    e chiama la funzione 'predici' del modulo corretto.
    """
    # 1. Carica configurazione
    if not os.path.exists(FILE_CONFIG_BEST_MODELS):
        print("Errore: File configurazione modelli non trovato. Esegui prima il training.")
        return None
    
    with open(FILE_CONFIG_BEST_MODELS, 'r') as f:
        config = json.load(f)
    
    modello_scelto = config.get(localita)
    
    if not modello_scelto:
        print(f"Errore: Nessun modello associato alla località {localita}")
        return None

    # 2. Recupera dati input
    temp_anno_prec = leggi_tmedia(localita, mese, giorno)
    
    # 3. Chiama la predizione del modulo specifico
    modulo = MAPPA_MODELLI[modello_scelto]
    predizione = modulo.predici(localita, anno, mese, giorno, temp_anno_prec)
    
    if predizione is not None:
        return predizione[0]
    return None


def predici_temperature_anno_citta(citta, anno):
    """
    Chiama la funzione 'predizione_annuale' del modello associato alla città.
    """
    # 1. Carica configurazione
    if not os.path.exists(FILE_CONFIG_BEST_MODELS):
        print("Errore: File configurazione modelli non trovato.")
        return {}
    
    with open(FILE_CONFIG_BEST_MODELS, 'r') as f:
        config = json.load(f)
    
    modello_scelto = config.get(citta)
    
    if not modello_scelto:
        print(f"Errore: Nessun modello associato alla località {citta}")
        return {}

    # 2. Chiama la funzione annuale del modulo specifico
    modulo = MAPPA_MODELLI[modello_scelto]
    risultato = modulo.predizione_annuale(citta, anno)
    
    return risultato

# Esempio di utilizzo (commentato per permettere l'importazione)
if __name__ == "__main__":
    # dataset_path = 'dati/dataset_climatico.csv' # Esempio
    # esegui_confronto_e_training(dataset_path, 'T_MEDIA', 2024)
    
    # Esempio predizione
    # val = predici_temperatura_localita('Bari', 2026, 8, 15)
    # print(f"Previsione Ferragosto 2026 Bari: {val}")
    pass