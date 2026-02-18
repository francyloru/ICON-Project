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

# Peso relativo della deviazione standard nello score composito.
# score = RMSE + ALPHA_STD_DEV * std_dev
# Con ALPHA = 1.0 i due termini hanno uguale peso.
# Abbassare ALPHA (es. 0.5) per privilegiare l'RMSE sulla variabilità.
ALPHA_STD_DEV = 0.3 # nel tempo la deviazione standard tende a "cancellarsi" a differenza dell'mrse quindi diamo più peso a quest'ultimo


def esegui_confronto_e_training(dataset, target_column, anno_test, citta_list):

    modelli_nomi = list(MAPPA_MODELLI.keys())

    # Dizionari per accumulare RMSE e std_dev separatamente
    # struttura: { 'modello': {'Bari': valore, 'Lecce': valore, ...} }
    risultati_rmse = {m: {} for m in modelli_nomi}
    risultati_std  = {m: {} for m in modelli_nomi}

    migliori_modelli = {}  # { 'Bari': 'xgboost', ... }

    print("=== INIZIO TRAINING E CONFRONTO MODELLI ===")

    for localita in citta_list:
        print(f"\n>>>> Elaborazione città: {localita}")
        best_score = float('inf')
        best_model_name = None

        for nome_modello, modulo in MAPPA_MODELLI.items():
            print(f"   > Training modello: {nome_modello}")

            # train_and_test ora restituisce (rmse, dev_standard)
            rmse, dev_standard = modulo.train_and_test(dataset, target_column, localita, anno_test)

            # Salviamo le due metriche nelle rispettive matrici
            risultati_rmse[nome_modello][localita] = rmse
            risultati_std[nome_modello][localita]  = dev_standard

            # ---------------------------------------------------------------
            # Score composito per la selezione del modello migliore:
            #   score = RMSE + ALPHA * std_dev
            #
            # Razionale: se due modelli hanno RMSE simile, viene preferito
            # quello con errori più consistenti (std_dev minore), cioè con
            # meno picchi di predizione errata. Il parametro ALPHA_STD_DEV
            # controlla quanto peso dare alla variabilità rispetto all'errore.
            # ---------------------------------------------------------------
            score_composito = rmse + ALPHA_STD_DEV * dev_standard

            if score_composito < best_score:
                best_score = score_composito
                best_model_name = nome_modello

            print("")

        migliori_modelli[localita] = best_model_name
        rmse_best = risultati_rmse[best_model_name][localita]
        std_best  = risultati_std[best_model_name][localita]
        print(f"   *** Miglior modello per {localita}: {best_model_name} "
              f"(RMSE: {rmse_best} | Std: {std_best} | Score: {rmse_best + ALPHA_STD_DEV * std_best:.3f}) ***")

    # ===================================================================
    # MATRICE DI CONFRONTO — unico CSV con RMSE e Std Dev affiancati
    # ===================================================================
    # Costruiamo un DataFrame con MultiIndex sulle colonne (Città, Metrica)
    # in modo che ogni colonna-città abbia sia RMSE che STD_DEV.

    # DataFrame RMSE (trasposto: righe = modelli, colonne = città)
    df_rmse = pd.DataFrame(risultati_rmse).T
    df_rmse.columns = [f"{c}_RMSE" for c in df_rmse.columns]

    # DataFrame Std Dev (stessa struttura)
    df_std = pd.DataFrame(risultati_std).T
    df_std.columns = [f"{c}_STD_DEV" for c in df_std.columns]

    # Interleaving: per ogni città mettiamo prima RMSE poi STD_DEV
    citta_ordinate = citta_list
    colonne_ordinate = []
    for citta in citta_ordinate:
        colonne_ordinate.append(f"{citta}_RMSE")
        colonne_ordinate.append(f"{citta}_STD_DEV")

    df_confronto = pd.concat([df_rmse, df_std], axis=1)[colonne_ordinate]

    # Visualizzazione a console (usiamo le due matrici separate per leggibilità)
    df_rmse_print = pd.DataFrame(risultati_rmse).T
    df_std_print  = pd.DataFrame(risultati_std).T

    print("\n" + "=" * 60)
    print(f"   MATRICE DI CONFRONTO — RMSE (anno test: {anno_test})")
    print("=" * 60)
    print(df_rmse_print)
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"   MATRICE DI CONFRONTO — STD DEV RESIDUI (anno test: {anno_test})")
    print("=" * 60)
    print(df_std_print)
    print("=" * 60)

    print("\nRIEPILOGO MIGLIORI MODELLI (score = RMSE + STD_DEV):")
    for citta in citta_list:
        modello_top = migliori_modelli[citta]
        r = risultati_rmse[modello_top][citta]
        s = risultati_std[modello_top][citta]
        print(f" - {citta}: {modello_top} (RMSE: {r} | Std: {s} | Score: {r + ALPHA_STD_DEV * s:.3f})")

    # Salvataggio associazione Città-Modello su JSON
    os.makedirs('modelli', exist_ok=True)
    with open(FILE_CONFIG_BEST_MODELS, 'w') as f:
        json.dump(migliori_modelli, f, indent=4)

    print(f"\nAssociazione modelli migliori salvata in '{FILE_CONFIG_BEST_MODELS}'")

    # Salvataggio CSV unico con entrambe le metriche affiancate per città
    os.makedirs('dati', exist_ok=True)
    df_confronto.to_csv('dati/confronto_metriche_modelli.csv', sep=';', decimal=',')
    print("Matrice di confronto (RMSE + Std Dev) salvata in 'dati/confronto_metriche_modelli.csv'")


def predici_temperatura_localita(localita, anno, mese, giorno):
    """
    Individua il modello migliore per la località, recupera la t_media anno prec
    e chiama la funzione 'predici' del modulo corretto.
    """
    if not os.path.exists(FILE_CONFIG_BEST_MODELS):
        print("Errore: File configurazione modelli non trovato. Esegui prima il training.")
        return None

    with open(FILE_CONFIG_BEST_MODELS, 'r') as f:
        config = json.load(f)

    modello_scelto = config.get(localita)

    if not modello_scelto:
        print(f"Errore: Nessun modello associato alla località {localita}")
        return None

    temp_anno_prec = leggi_tmedia(localita, mese, giorno)

    modulo = MAPPA_MODELLI[modello_scelto]
    predizione = modulo.predici(localita, anno, mese, giorno, temp_anno_prec)

    if predizione is not None:
        return predizione[0]
    return None


def predici_temperature_anno_citta(citta, anno):
    if not os.path.exists(FILE_CONFIG_BEST_MODELS):
        print("Errore: File configurazione modelli non trovato.")
        return {}

    with open(FILE_CONFIG_BEST_MODELS, 'r') as f:
        config = json.load(f)

    modello_scelto = config.get(citta)

    if not modello_scelto:
        print(f"Errore: Nessun modello associato alla località {citta}")
        return {}

    modulo = MAPPA_MODELLI[modello_scelto]
    risultato = modulo.predizione_annuale(citta, anno)

    return risultato