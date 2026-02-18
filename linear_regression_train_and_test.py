import pandas as pd
import numpy as np
import joblib
import os
import math
from datetime import datetime
import calendar
from dati.gestore import leggi_tmedia

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error


def train_and_test(dataset, target_column, localita, anno_test):
    df = pd.read_csv(dataset, sep=';')
    features = ['ANNO', 'SIN_GIORNO', 'COS_GIORNO', 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE']
    df_clean = df.dropna(subset=['LOCALITA'] + features + [target_column])

    train_df = df_clean[(df_clean['ANNO'] < anno_test) & (df_clean['LOCALITA'] == localita)]
    test_df = df_clean[(df_clean['ANNO'] == anno_test) & (df_clean['LOCALITA'] == localita)]

    X_train = train_df[features]
    y_train = train_df[target_column]
    X_test = test_df[features]
    y_test = test_df[target_column]

    print(f"   - [LinearReg] Training sugli anni antecedenti al {anno_test} per {localita}")

    param_grid = {
        'fit_intercept': [True, False]
    }

    lr_model = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(lr_model, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)

    print("   - Addestramento iniziato.")
    grid_search.fit(X_train, y_train)
    print("   - Addestramento completato.")

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

    print(f"\n   - Risultati Test (RMSE): {rmse:.3f} °C")
    print(f"   - Deviazione Standard residui: {dev_standard:.3f} °C")
    print(f"   - Bias medio: {float(np.mean(residui)):.3f} °C")

    # Salvataggio Parametri
    results_df = pd.DataFrame(grid_search.cv_results_)
    colonne_interessanti = [col for col in results_df.columns if 'param_' in col or 'mean_test_score' in col]
    results_df = results_df[colonne_interessanti]
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df.rename(columns={'mean_test_score': 'RMSE_medio_CV'})
    results_df[f'RMSE_Test_{anno_test}'] = np.nan
    results_df[f'STD_DEV_Test_{anno_test}'] = np.nan
    results_df.loc[grid_search.best_index_, f'RMSE_Test_{anno_test}'] = rmse
    results_df.loc[grid_search.best_index_, f'STD_DEV_Test_{anno_test}'] = dev_standard
    results_df = results_df.round(5)

    os.makedirs('dati/parametri', exist_ok=True)
    results_df.to_csv(f'dati/parametri/parametri_linear_regression_{localita}.csv', index=False, sep=';', decimal=',')

    # Salvataggio Predizioni
    os.makedirs('dati/risultati_dei_modelli', exist_ok=True)
    risultati = test_df.copy()
    risultati["PRED_LINREG"] = pred_test
    risultati["PRED_LINREG"] = risultati["PRED_LINREG"].round(2)
    risultati.to_csv(f'./dati/risultati_dei_modelli/predizioni_linear_regression_{localita}.csv', index=False, sep=';', decimal=',')

    # Retrain
    print("\n   -> Retraining finale.")
    final_train_df = df_clean[(df_clean['ANNO'] <= anno_test) & (df_clean['LOCALITA'] == localita)]
    final_model = LinearRegression(**grid_search.best_params_)
    final_model.fit(final_train_df[features], final_train_df[target_column])

    os.makedirs('modelli', exist_ok=True)
    joblib.dump(final_model, f'modelli/modello_linear_regression_{localita}.pkl')
    print(f"   --> Modello salvato: 'modello_linear_regression_{localita}.pkl'")

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
        modello = joblib.load(f'modelli/modello_linear_regression_{localita}.pkl')
    except FileNotFoundError:
        print("Errore: Modello LR non trovato.")
        return

    try:
        data_obj = datetime(anno, mese, giorno)
        giorno_anno = data_obj.timetuple().tm_yday
        sin_giorno = round(math.sin(2 * math.pi * giorno_anno / 365.25), 5)
        cos_giorno = round(math.cos(2 * math.pi * giorno_anno / 365.25), 5)
    except ValueError:
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
            if valore is not None:
                risultato[(mese, giorno)] = valore
    return risultato