import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime
import xgboost as xgb  # Importiamo XGBoost
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error

def train_and_test(dataset, target_column):
    # 1. CARICAMENTO
    df = pd.read_csv(dataset, sep=';')

    # 2. SELEZIONE DELLE FEATURES RICHIESTE
    # Il modello userà solo queste 4 colonne come input
    features = ['ANNO', 'SIN_GIORNO', 'COS_GIORNO', 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE']
    
    # Pulizia: rimuoviamo righe dove i dati storici o il target sono NULL
    df_clean = df.dropna(subset=features + [target_column])

    X = df_clean[features]
    y = df_clean[target_column]

    # 3. SPLIT DATI
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Salvataggio struttura colonne per l'uso futuro
    # pd.DataFrame(X.columns.tolist(), columns=['nome_colonna']).to_csv('dati/struttura_colonne_xgb.csv', index=False, sep=';')

    # 4. PARAMETRI XGBOOST PER GRID SEARCH
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # Inizializzazione XGBRegressor
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)

    print("  - Addestramento XGBoost con K-Fold Iniziato...")
    grid_search.fit(X_train, y_train)
    print("  - Addestramento Terminato.")

    # 5. SALVATAGGIO E VALUTAZIONE
    best_model = grid_search.best_estimator_
    mae_finale = mean_absolute_error(y_test, best_model.predict(X_test))
    
    print(f"  - Migliori parametri: {grid_search.best_params_}")
    print(f"  - Errore medio (MAE) sul test set: {mae_finale:.2f} °C")

    joblib.dump(best_model, 'modello_temperatura_xgb.pkl')
    print("  - Modello salvato come 'modello_temperatura_xgb.pkl'")

def usa_modello():
    try:
        modello = joblib.load('modello_temperatura_xgb.pkl')
    except FileNotFoundError:
        print("Errore: Modello non trovato. Eseguire prima il training.")
        return

    print("\n--- PREVISIONE TEMPERATURA MEDIA (XGBOOST) ---")
    
    # 1. INPUT UTENTE (Giorno e Mese richiesti esplicitamente)
    anno = int(input("Anno della previsione: "))
    mese = int(input("Mese (1-12): "))
    giorno = int(input("Giorno (1-31): "))
    temp_anno_prec = float(input("Temperatura media dello stesso giorno anno precedente (°C): "))

    # 2. CALCOLO CICLICITÀ (SIN/COS)
    try:
        data_obj = datetime(anno, mese, giorno)
        giorno_anno = data_obj.timetuple().tm_yday
        giorni_totali = 365.25
        
        sin_giorno = round(math.sin(2 * math.pi * giorno_anno / giorni_totali), 5)
        cos_giorno = round(math.cos(2 * math.pi * giorno_anno / giorni_totali), 5)
    except ValueError:
        print("Data non valida!")
        return

    # 3. CREAZIONE DATAFRAME DI INPUT (Deve avere lo stesso ordine del training)
    # Ordine: ['ANNO', 'SIN_GIORNO', 'COS_GIORNO', 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE']
    input_data = pd.DataFrame([[anno, sin_giorno, cos_giorno, temp_anno_prec]], 
                               columns=['ANNO', 'SIN_GIORNO', 'COS_GIORNO', 'TEMPERATURA_MEDIA_ANNO_PRECEDENTE'])

    # 4. PREDIZIONE
    previsione = modello.predict(input_data)
    
    print("\n" + "="*40)
    print(f"DATA: {giorno}/{mese}/{anno}")
    print(f"PREVISIONE TEMPERATURA MEDIA: {previsione[0]:.2f} °C")
    print("="*40 + "\n")

# Per testare:
# train_and_test_xgb('dati/dataset_meteo_unificato.csv')
# usa_modello_xgb()