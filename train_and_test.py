import pandas as pd
# import numpy as np
import joblib # Per salvare il modello
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error

def train_and_test(dataset):
    # 1. CARICAMENTO
    df = pd.read_csv(dataset, sep=';')

    # 2. PREPARAZIONE DATI DINAMICA
    # Se ci sono colonne di testo (come LOCALITA o FENOMENI), le trasformiamo in numeri 
    # usando il Factorize (Bari=0, Roma=1, ecc.) per non perdere l'informazione della città
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]

    # Definiamo Target e Features
    y = df['PIOGGIA mm']
    # Prendiamo TUTTO tranne la pioggia mm (che è la risposta che cerchiamo)
    X = df.drop(columns=['PIOGGIA mm'])

    # 3. K-FOLD CROSS VALIDATION TRAMITE GRID SEARCH
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Ottieni la lista delle colonne
    colonne_finali = X.columns.tolist()
    # Creiamo un piccolo DataFrame con una sola colonna chiamata 'nome_colonna'
    df_struttura = pd.DataFrame(colonne_finali, columns=['nome_colonna'])
    # Salvataggio in CSV
    df_struttura.to_csv('dati/struttura_colonne.csv', index=False, sep=';')
    print("  - La struttura della tabella è stata salvata nel file 'dati/struttura_colonne.csv'")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4] # Un altro parametro utile per la stabilità
    }

    # cv=5 esegue la K-Fold Cross Validation su 5 blocchi
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                            param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    print("  - Addestramento con K-Fold Iniziato.")
    grid_search.fit(X_train, y_train)
    print("  - Addestramento con K-Fold Terminato.")

    # 1. Trasformiamo i risultati in un DataFrame
    risultati = pd.DataFrame(grid_search.cv_results_)
    # 2. Selezioniamo solo le colonne che contano davvero
    # - 'params': la combinazione di iperparametri
    # - 'mean_test_score': il MAE medio (negativo)
    # - 'std_test_score': quanto è variato l'errore tra i vari fold (indica stabilità)
    # - 'rank_test_score': la posizione nella classifica dei migliori
    tabella_pulita = risultati[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    # 4. Esportazione in CSV
    tabella_pulita.to_csv('classifica_parametri_ottimali.csv', index=False, sep=';')

    print("  - E' stato creato il file 'classifica_parametri_ottimali.csv'.")
    # 4. SALVATAGGIO DEL MODELLO
    best_model = grid_search.best_estimator_


    print(f"  - I migliori parametri per il Modello sono: {grid_search.best_params_}")

    # --- AGGIUNTA: VALUTAZIONE FINALE SUL TEST SET ---
    y_pred = best_model.predict(X_test)
    mae_finale = mean_absolute_error(y_test, y_pred)
    
    print(f"  - TEST SET EVALUATION: Errore medio reale (MAE) su dati mai visti: {mae_finale:.2f} mm")
    # ------------------------------------------------


    joblib.dump(best_model, 'modello_previsione_pioggia.pkl')
    print("  - Il Modello è stato salvato come 'modello_previsione_pioggia.pkl' .\n")


def usa_modello():
    try:
        modello = joblib.load('modello_previsione_pioggia.pkl')
        df_struttura = pd.read_csv('dati/struttura_colonne.csv', sep=';')
        nomi_colonne = df_struttura['nome_colonna'].tolist()
    except FileNotFoundError:
        print("Errore: File necessari mancanti.")
        return

    print("\n--- INSERIMENTO DATI METEO COMPLETI ---")
    
    # --- INPUT DATI CATEGORICI ---
    loc = input("Località (es. Bari): ")
    fenomeno = input("Fenomeno (soleggiato, pioggia, nebbia, pioggia temporale): ").lower().strip()
    mese = input("Mese (1-12): ")
    
    # --- INPUT DATI NUMERICI (TUTTI) ---
    dati = {}
    dati['ANNO'] = int(input("Anno: "))
    dati['GIORNO'] = int(input("Giorno: "))
    dati['TMEDIA °C'] = float(input("Temp Media (°C): "))
    dati['TMIN °C'] = float(input("Temp Minima (°C): "))
    dati['TMAX °C'] = float(input("Temp Massima (°C): "))
    dati['PUNTORUGIADA °C'] = float(input("Punto Rugiada (°C): "))
    dati['UMIDITA %'] = float(input("Umidità (%): "))
    dati['VISIBILITA km'] = float(input("Visibilità (km): "))
    dati['VENTOMEDIA km/h'] = float(input("Vento Medio (km/h): "))
    dati['VENTOMAX km/h'] = float(input("Vento Max (km/h): "))
    dati['RAFFICA km/h'] = float(input("Raffica (km/h): "))
    dati['PRESSIONESLM mb'] = float(input("Pressione SLM (mb): "))
    dati['PRESSIONEMEDIA mb'] = float(input("Pressione Media (mb): "))

    # 4. CREAZIONE DATAFRAME DI INPUT
    input_df = pd.DataFrame(0.0, index=[0], columns=nomi_colonne)

    # 5. RIEMPIMENTO DATI NUMERICI
    for campo, valore in dati.items():
        if campo in input_df.columns:
            input_df.loc[0, campo] = valore

    # 6. RIEMPIMENTO ONE-HOT (Mese, Località, Fenomeni)
    col_mese = f"Mese_{mese}"
    col_loc = f"LOCALITA_{loc}"
    col_fen = f"FENOMENI_{fenomeno}"

    for col in [col_mese, col_loc, col_fen]:
        if col in input_df.columns:
            input_df.loc[0, col] = 1.0
        else:
            print(f"Nota: La categoria '{col}' non era presente nell'addestramento (impostata a 0).")

    # 7. PREDIZIONE
    previsione = modello.predict(input_df)
    
    print("\n" + "="*40)
    print(f"RISULTATO ANALISI: {previsione[0]:.2f} mm di pioggia")
    print("="*40 + "\n")
    
    # Restituiamo il valore per un eventuale uso in Prolog
    return previsione[0], loc, fenomeno