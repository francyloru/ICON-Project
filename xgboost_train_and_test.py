import pandas as pd
import numpy as np
import joblib
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
from dati.gestore import leggi_tmedia

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error


def train_and_test(dataset, target_column, localita, anno_test):
    # ===============================
    # 1. CARICAMENTO DATI
    # ===============================
    df = pd.read_csv(dataset, sep=';')

    features = [
        'ANNO',
        'SIN_GIORNO',
        'COS_GIORNO',
        'TEMPERATURA_MEDIA_ANNO_PRECEDENTE'
    ]

    df_clean = df.dropna(subset=['LOCALITA'] + features + [target_column])

    # ===============================
    # 2. SPLIT TEMPORALE
    # ===============================
    train_df = df_clean[(df_clean['ANNO'] < anno_test) & (df_clean['LOCALITA'] == localita)]
    test_df = df_clean[(df_clean['ANNO'] == anno_test) & (df_clean['LOCALITA'] == localita)]

    X_train = train_df[features]
    y_train = train_df[target_column]

    X_test = test_df[features]
    y_test = test_df[target_column]

    print(f"   - Training sugli anni antecedenti al {anno_test} per {localita} (Train size: {len(X_train)})")
    print(f"   - Test sull'anno {anno_test} (Test size: {len(X_test)})")

    # ===============================
    # 3. GRID SEARCH
    # ===============================
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    print("   - [XGBoost] Addestramento con TimeSeriesSplit iniziato.")
    grid_search.fit(X_train, y_train)
    print("   - [XGBoost] Addestramento completato.")

    best_model = grid_search.best_estimator_
    pred_test = best_model.predict(X_test)

    rmse = root_mean_squared_error(y_test, pred_test)

    # ===============================
    # CALCOLO DEVIAZIONE STANDARD DEI RESIDUI
    # ===============================
    # La deviazione standard degli errori misura la dispersione dei residui
    # attorno al loro valor medio. Complementa l'RMSE fornendo info sulla
    # variabilità delle predizioni (RMSE² = bias² + std_dev²).
    residui = np.array(y_test) - pred_test
    dev_standard = float(np.std(residui))

    print("\n   - Risultati Test:")
    print(f"            - Migliori parametri: {grid_search.best_params_}")
    print(f"            - RMSE per l'anno {anno_test}: {rmse:.3f} °C")
    print(f"            - Deviazione Standard residui: {dev_standard:.3f} °C")
    print(f"            - Bias medio: {float(np.mean(residui)):.3f} °C")

    # -----------------------------------------------------------------------
    # Salvataggio parametri
    results_df = pd.DataFrame(grid_search.cv_results_)
    colonne_interessanti = [col for col in results_df.columns if 'param_' in col or 'mean_test_score' in col or 'rank_test_score' in col]
    results_df = results_df[colonne_interessanti]
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df.rename(columns={'mean_test_score': 'RMSE_medio_CV'})
    results_df[f'RMSE_Test_{anno_test}'] = np.nan
    results_df[f'STD_DEV_Test_{anno_test}'] = np.nan
    best_index = grid_search.best_index_
    results_df.loc[best_index, f'RMSE_Test_{anno_test}'] = rmse
    results_df.loc[best_index, f'STD_DEV_Test_{anno_test}'] = dev_standard
    results_df = results_df.round(5)

    os.makedirs('dati/parametri', exist_ok=True)
    results_df.to_csv(
        f'dati/parametri/parametri_xgboost_{localita}.csv',
        index=False,
        sep=';',
        decimal=','
    )
    print(f"   - Salvati gli iperparametri in 'parametri_xgboost_{localita}.csv'")
    # -----------------------------------------------------------------------

    # ===============================
    # 6. SALVATAGGIO PREDIZIONI
    # ===============================
    os.makedirs('dati/risultati_dei_modelli', exist_ok=True)
    risultati = test_df.copy()
    risultati["PRED_XGBOOST"] = pred_test
    risultati["PRED_XGBOOST"] = risultati["PRED_XGBOOST"].round(2)

    risultati.to_csv(
        f'./dati/risultati_dei_modelli/predizioni_xgboost_{localita}.csv',
        index=False,
        sep=';',
        decimal=','
    )

    # ===============================
    # 8. RETRAIN FINALE
    # ===============================
    print("\n   -> Iniziato il Retraining sull'intero dataset.")

    final_train_df = df_clean[
        (df_clean['ANNO'] <= anno_test) & (df_clean['LOCALITA'] == localita)
    ]

    X_final = final_train_df[features]
    y_final = final_train_df[target_column]

    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **grid_search.best_params_
    )

    final_model.fit(X_final, y_final)

    os.makedirs('modelli', exist_ok=True)
    joblib.dump(final_model, f'modelli/modello_xgboost_{localita}.pkl')
    print(f"   --> Modello salvato come 'modello_xgboost_{localita}.pkl'")

    # Restituisce sia RMSE che deviazione standard dei residui
    return round(rmse, 3), round(dev_standard, 3)


def usa_modello(localita):
    anno = 2026
    print(f"  - Località: {localita}")
    print(f"  - Anno della previsione: {anno}")

    mese = int(input("  - Mese (1-12): "))
    giorno = int(input("  - Giorno (1-31): "))
    temp_anno_prec = leggi_tmedia(localita, mese, giorno)
    print(f"  - Temperatura media dello stesso giorno anno precedente (°C): {temp_anno_prec}")

    previsione = predici(localita, anno, mese, giorno, temp_anno_prec)

    print("\n" + "=" * 42)
    print(f"  DATA: {giorno}/{mese}/{anno}")
    print(f"  PREVISIONE TEMPERATURA MEDIA: {previsione:.2f} °C")
    print("=" * 42 + "\n")


def predici(localita, anno, mese, giorno, temp_anno_prec):
    try:
        modello = joblib.load(f'modelli/modello_xgboost_{localita}.pkl')
    except FileNotFoundError:
        print("Errore: Modello non trovato. Eseguire prima il training.")
        return

    try:
        data_obj = datetime(anno, mese, giorno)
        giorno_anno = data_obj.timetuple().tm_yday
        giorni_totali = 365.25
        sin_giorno = round(math.sin(2 * math.pi * giorno_anno / giorni_totali), 5)
        cos_giorno = round(math.cos(2 * math.pi * giorno_anno / giorni_totali), 5)
    except ValueError:
        print("Data non valida!")
        return

    input_data = pd.DataFrame([[anno, sin_giorno, cos_giorno, float(temp_anno_prec)]],
                               columns=['ANNO', 'SIN_GIORNO', 'COS_GIORNO', 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE'])

    return float(modello.predict(input_data)[0])


def predizione_annuale(localita, anno):
    risultato = {}
    for mese in range(1, 13):
        giorni_nel_mese = calendar.monthrange(anno, mese)[1]
        for giorno in range(1, giorni_nel_mese + 1):
            valore = predici(localita, anno, mese, giorno, leggi_tmedia(localita, mese, giorno))
            risultato[(mese, giorno)] = valore
    return risultato