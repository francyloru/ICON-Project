# Sistema di Supporto alla Pianificazione Agricola tramite Predizione Meteo e Ricerca A*

Questo progetto implementa un **Knowledge-Based System (KBS)** per la pianificazione ottimale delle colture in serre distribuite in diverse città.

L'obiettivo è determinare **in quale città e in quale periodo coltivare ogni pianta** in modo da **minimizzare il costo energetico di climatizzazione delle serre**, sfruttando:

- **Machine Learning** per la predizione della temperatura giornaliera
- **Ricerca informata (A*)** per lo scheduling ottimale delle colture


---
# Architettura del Sistema

Il sistema è composto da due moduli principali di Intelligenza Artificiale:

## 1. Predizione meteorologica (Machine Learning)

Questo modulo stima la **temperatura media giornaliera** per ogni città nell'anno target.

Vengono addestrati tre modelli di regressione:

- Linear Regression
- Random Forest
- XGBoost

Per ogni città viene selezionato automaticamente **il modello con le migliori prestazioni sul set di test**, ovvero 
quello che minimizza l'errore sulla predizione (quello che dopo indichiamo come **score**).

## 2. Scheduling delle colture (Ricerca A*)

Una volta ottenute le temperature previste, il sistema calcola la **schedulazione ottimale delle colture** nelle serre, indica cioè
la città e il giorno preciso in cui piantare ciascuna coltura per minimizzare il **costo terminco**.

Il costo energetico è definito come:

<code>Costo = |T_predetta − T_ideale|</code>

cioè la quantità di energia necessaria per mantenere la serra alla temperatura ottimale.

---

# Pipeline del Progetto

L'intero workflow è eseguibile tramite interfaccia a riga di comando:

## 1. Lettura, formalizzazione del dataset e feature engineering

<code>python main.py --new_dataset</code>

Questo comando:

- Aggrega i dati meteorologici storici presenti nella cartella <code>dati/dati_meteo_separati_csv</code> e che sono
  stati prelevati precedentemente dal sito <a href="https://www.ilmeteo.it/portale/archivio-meteo/">ilmeteo.it</a>
- Costruisce il dataset unificato
- Genera le feature utilizzate dai modelli

Le feature principali utilizzate per l'addestramento sono:

<code> X = [ANNO, SIN_GIORNO, COS_GIORNO, TEMPERATURA_MEDIA_ANNO_PRECEDENTE] </code>

### Nota tecnica
Per rappresentare la **stagionalità della temperatura** si utilizza una codifica ciclica del giorno dell'anno:
- <code> sin(2πJ / 365)</code>
- <code> cos(2πJ / 365)</code>

## 2. Addestramento e selezione dei modelli

<code> python main.py --find_models </code>

Per ogni città vengono addestrati:

- Linear Regression
- Random Forest
- XGBoost

La selezione degli iperparametri avviene tramite:

<code> GridSearchCV + TimeSeriesSplit </code>

### Nota tecnica

`TimeSeriesSplit` mantiene l'ordine temporale dei dati ed evita **data leakage**, che si verificherebbe se informazioni future fossero utilizzate nel training.

Il modello migliore è quello che minimizza lo score:

<code> score = RMSE + 0.3 * σ_residui </code>

che penalizza sia l'errore medio sia la variabilità delle predizioni.

## 3. Scheduling ottimale con A*

<code> python main.py --find_scheduling </code>


L'algoritmo **A\*** determina per ogni coltura:

- la città
- il giorno in cui piantare la specifica coltura


Vincoli:

- ogni città possiede **una sola serra**
- ogni serra può ospitare **una coltura alla volta**

---

# Modellazione del Problema di Ricerca

## 1. Stato

Uno stato è rappresentato come:

<code> (disp, piante_rimanenti)</code>

dove:

- `disp` indica il primo giorno disponibile per ogni serra
- `piante_rimanenti` rappresenta l'insieme delle colture non ancora pianificate


## 2. Azione

Un'azione consiste nell'assegnare una coltura `p` a una città `c` con giorno di inizio `d`.

Dopo l'assegnazione:


<code>disp[c] = d + durata_coltura</code>


## 3. Funzione di costo

Il costo di coltivazione è definito come:


<code> cost = Σ |T(c,k) − T_ideale(p)| </code>


ovvero la somma delle differenze tra temperatura prevista e temperatura ideale lungo tutto il ciclo vegetativo.


# Euristica di A*

L'euristica utilizzata è:


<code> h(s) = somma dei costi minimi possibili per ogni coltura rimanente </code>


ignorando temporaneamente:

- conflitti tra colture
- disponibilità delle serre

Proprietà dell'euristica:

- **Ammissibile** → non sovrastima mai il costo reale
- **Consistente** → A* non riesplora stati già espansi

Questo garantisce **ottimalità della soluzione trovata**.

---

# Ottimizzazione delle Prestazioni

Prima dell'esecuzione di A*, il sistema costruisce una **lookup table dei costi**:


<code> COSTI_PRECALCOLATI[pianta][città][giorno_start] </code>


Questo consente di ottenere il costo di ogni assegnazione in **tempo O(1)** durante la ricerca.

---

# Configurazione

Il sistema è completamente configurabile tramite appositi file JSON:

## Colture

<code>colture.json</code>

Contiene:

- temperatura ideale di crescita
- durata del ciclo vegetativo
<i>(dati presi da fonti facilmente reperibili online)</i>

## Parametri


<code>parametri.json</code>


Contiene:

- Le città considerate
- L'anno target (cioè quello in cui effettuare la predizione

---

# Requisiti

Python **3.x**

Installazione delle dipendenze:


<code> pip install pandas scikit-learn xgboost </code>


# Possibili Sviluppi Futuri

Possibili estensioni del sistema includono:

- euristiche più informative per ridurre i nodi esplorati
- integrazione di previsioni meteorologiche ufficiali
- introduzione di vincoli agronomici più realistici (rotazione delle colture, irrigazione, costi di setup)


