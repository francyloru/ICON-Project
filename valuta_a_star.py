"""
benchmark_a_star.py
====================
Valuta le prestazioni dell'algoritmo A* al crescere del numero di colture
e città, producendo una tabella di confronto con le seguenti metriche:
  - Tempo di esecuzione (secondi)
  - Nodi esplorati (stati aggiunti a visited_states)
  - Nodi generati (stati inseriti nell'open set, inclusi i duplicati)
  - Energia della soluzione trovata
  - Lower bound euristico iniziale
  - Gap% = (energia - lower_bound) / lower_bound * 100
  - Costo medio per nodo esplorato (tempo / nodi_esplorati)

UTILIZZO:
    python benchmark_a_star.py

Il file genera:
  - Stampa a console della tabella formattata
  - 'benchmark_risultati.csv' con i dati grezzi
"""

import heapq
import time
import csv
import itertools
from datetime import date, timedelta
import json

# ---------------------------------------------------------------------------
# Import del modulo principale: usiamo i dati già caricati (TEMPERATURE,
# COSTI_PRECALCOLATI, COLTURE) per non duplicare la logica di caricamento.
# ---------------------------------------------------------------------------
import cerca_con_a_star as astar


# =============================================================================
# VERSIONE STRUMENTATA DI run_a_star
# Identica all'originale ma:
#   1. Accetta colture_subset e citta_subset come parametri (niente globals)
#   2. Restituisce anche nodi_esplorati e nodi_generati
# =============================================================================

def _run_a_star_strumentato(colture_subset: dict, citta_subset: list, anno_target: int):
    """
    Versione di run_a_star che lavora su sottoinsiemi di colture/città
    e restituisce le metriche di analisi oltre alla soluzione.

    Returns
    -------
    energia       : float  – costo totale della soluzione (None se fallisce)
    piano         : list   – lista di azioni della soluzione
    nodi_esplorati: int    – stati effettivamente espansi (visited_states)
    nodi_generati : int    – stati inseriti nella coda (compresi i duplicati scartati)
    """

    def _calcola_euristica_locale(piante_rim, citta):
        h = 0
        for p in piante_rim:
            min_g = float('inf')
            for c in citta:
                val = min(astar.COSTI_PRECALCOLATI[p][c])
                if val < min_g:
                    min_g = val
            if min_g != float('inf'):
                h += min_g
        return h

    def _trova_miglior_start_locale(pianta, citta, giorno_min):
        costi = astar.COSTI_PRECALCOLATI[pianta][citta]
        best_day, min_cost = -1, float('inf')
        for d in range(giorno_min, len(costi)):
            if costi[d] < min_cost:
                min_cost = costi[d]
                best_day = d
        return best_day, min_cost

    piante_init = tuple(sorted(colture_subset.keys()))
    disp_init   = tuple([0] * len(citta_subset))

    start_h    = _calcola_euristica_locale(piante_init, citta_subset)
    lower_bound = start_h   # salviamo per il calcolo del gap

    counter       = 0
    nodi_esplorati = 0
    nodi_generati  = 1   # contiamo il nodo iniziale

    open_set      = [(start_h, 0, counter, disp_init, piante_init, [])]
    visited_states = set()

    while open_set:
        f, g, _, disp, piante, storia = heapq.heappop(open_set)

        # GOAL STATE
        if not piante:
            return g, storia, nodi_esplorati, nodi_generati, lower_bound

        # Pruning
        state_sig = (disp, piante)
        if state_sig in visited_states:
            continue
        visited_states.add(state_sig)
        nodi_esplorati += 1

        # Espansione
        for idx_p, pianta_target in enumerate(piante):
            restanti = piante[:idx_p] + piante[idx_p + 1:]

            for i, citta in enumerate(citta_subset):
                giorno_libero = disp[i]
                best_start, costo_energia = _trova_miglior_start_locale(
                    pianta_target, citta, giorno_libero
                )

                if best_start != -1:
                    new_g = g + costo_energia
                    new_h = _calcola_euristica_locale(restanti, citta_subset)
                    new_f = new_g + new_h

                    durata    = colture_subset[pianta_target]['durata']
                    new_disp  = list(disp)
                    new_disp[i] = best_start + durata

                    new_action = {
                        'citta'  : citta,
                        'pianta' : pianta_target,
                        'start'  : best_start,
                        'end'    : best_start + durata,
                        'costo'  : costo_energia
                    }

                    counter += 1
                    nodi_generati += 1

                    heapq.heappush(open_set, (
                        new_f, new_g, counter,
                        tuple(new_disp), restanti,
                        storia + [new_action]
                    ))

    return None, None, nodi_esplorati, nodi_generati, lower_bound


# =============================================================================
# GENERAZIONE SCENARI CRESCENTI
# Strategia: fissiamo il numero di città (1 → max) e per ciascuno facciamo
# crescere il numero di colture (1 → max). Otteniamo una griglia completa.
# =============================================================================

def _genera_scenari(tutte_le_citta: list, tutte_le_colture: dict):
    """
    Restituisce una lista di scenari ordinati per complessità crescente.
    Ogni scenario è un dict con 'citta' (list) e 'colture' (dict).

    Ordine: prima aumentiamo le colture a parità di città (riga per riga),
    poi passiamo alla riga successiva con una città in più.
    Questo rende la progressione della tabella intuitiva da leggere.
    """
    nomi_colture = list(tutte_le_colture.keys())
    scenari = []

    for n_citta in range(1, len(tutte_le_citta) + 1):
        citta_subset = tutte_le_citta[:n_citta]

        for n_colture in range(1, len(nomi_colture) + 1):
            colture_subset_nomi = nomi_colture[:n_colture]
            colture_subset = {k: tutte_le_colture[k] for k in colture_subset_nomi}

            scenari.append({
                'n_citta'  : n_citta,
                'n_colture': n_colture,
                'citta'    : citta_subset,
                'colture'  : colture_subset
            })

    return scenari


# =============================================================================
# FUNZIONE PRINCIPALE DI BENCHMARK
# =============================================================================

def esegui_benchmark(anno_target: int, tutte_le_citta: list, output_csv: str = 'dati/benchmark_risultati.csv'):
    """
    Esegue il benchmark su tutti gli scenari crescenti e stampa + salva la tabella.

    Parameters
    ----------
    anno_target    : anno di riferimento per le previsioni meteo
    tutte_le_citta : lista completa di città disponibili (nell'ordine desiderato)
    output_csv     : nome del file CSV di output
    """

    # -------------------------------------------------------------------------
    # 1. Assicuriamoci che i dati meteo e i costi siano caricati per TUTTE
    #    le città e colture prima di iniziare i test.
    #    Se sono già stati caricati dal modulo principale, questo è no-op.
    # -------------------------------------------------------------------------
    print("="*55)
    print("     BENCHMARK A* — Caricamento dati completi")
    print("="*55)

    citta_da_caricare = [c for c in tutte_le_citta if c not in astar.TEMPERATURE]
    if citta_da_caricare:
        astar.carica_dati_meteo(anno_target, citta_da_caricare)
    
    astar.precalcola_costi(anno_target, tutte_le_citta)

    # -------------------------------------------------------------------------
    # 2. Genera gli scenari crescenti
    # -------------------------------------------------------------------------
    scenari = _genera_scenari(tutte_le_citta, astar.COLTURE)

    print(f"\nScenari da testare: {len(scenari)}")
    print(f"Colture disponibili: {list(astar.COLTURE.keys())}")
    print(f"Città disponibili:   {tutte_le_citta}\n")

    # -------------------------------------------------------------------------
    # 3. Esegui ogni scenario e raccogli i risultati
    # -------------------------------------------------------------------------
    intestazioni = [
        'Città (N)', 'Colture (M)', 'Nomi Colture', 'Nomi Città',
        'Tempo (s)', 'Nodi Esplorati', 'Nodi Generati',
        'Energia', 'Lower Bound', 'Gap (%)',
        'Tempo/Nodo (ms)', 'Rapporto Gen/Esp'
    ]

    righe = []

    sep = "-" * 95
    header = (f"{'C':>4} {'P':>4} | {'Tempo(s)':>9} {'N.Esp.':>10} {'N.Gen.':>10} "
              f"{'Energia':>10} {'L.Bound':>10} {'Gap%':>7} {'ms/nodo':>9} {'Gen/Esp':>8}")

    print(sep)
    print(header)
    print(sep)

    for s in scenari:
        t_start = time.perf_counter()

        energia, piano, n_esp, n_gen, lower_bound = _run_a_star_strumentato(
            s['colture'], s['citta'], anno_target
        )

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        if energia is None:
            # Nessuna soluzione (non dovrebbe accadere con dati validi)
            gap        = float('nan')
            ms_per_nodo = float('nan')
            gen_esp_ratio = float('nan')
            energia_str = 'N/A'
            lb_str      = 'N/A'
            gap_str     = 'N/A'
            ms_str      = 'N/A'
            ratio_str   = 'N/A'
        else:
            gap           = ((energia - lower_bound) / lower_bound * 100) if lower_bound > 0 else 0.0
            ms_per_nodo   = (elapsed / n_esp * 1000) if n_esp > 0 else 0.0
            gen_esp_ratio = (n_gen / n_esp) if n_esp > 0 else float('inf')
            energia_str   = f"{energia:.1f}"
            lb_str        = f"{lower_bound:.1f}"
            gap_str       = f"{gap:.2f}"
            ms_str        = f"{ms_per_nodo:.4f}"
            ratio_str     = f"{gen_esp_ratio:.2f}"

        # Stampa riga console
        print(f"{s['n_citta']:>4} {s['n_colture']:>4} | "
              f"{elapsed:>9.4f} {n_esp:>10} {n_gen:>10} "
              f"{energia_str:>10} {lb_str:>10} {gap_str:>7} "
              f"{ms_str:>9} {ratio_str:>8}")

        # Riga per il CSV
        righe.append([
            s['n_citta'],
            s['n_colture'],
            ' | '.join(s['colture'].keys()),
            ' | '.join(s['citta']),
            round(elapsed, 5),
            n_esp,
            n_gen,
            round(energia, 2) if energia is not None else 'N/A',
            round(lower_bound, 2),
            round(gap, 3) if energia is not None else 'N/A',
            round(ms_per_nodo, 5) if energia is not None else 'N/A',
            round(gen_esp_ratio, 3) if energia is not None else 'N/A'
        ])

    print(sep)
    print("\nLegenda colonne:")
    print("  C        = numero di città")
    print("  P        = numero di colture (piante)")
    print("  N.Esp.   = nodi effettivamente espansi (visited_states)")
    print("  N.Gen.   = nodi inseriti nella coda (inclusi duplicati)")
    print("  L.Bound  = lower bound euristico calcolato sullo stato iniziale")
    print("  Gap%     = quanto la soluzione dista dal lower bound (ideale: vicino a 0%)")
    print("  ms/nodo  = millisecondi spesi per ogni nodo esplorato (deve restare costante)")
    print("  Gen/Esp  = quanti nodi vengono generati per ogni nodo esplorato (qualità euristica)")

    # -------------------------------------------------------------------------
    # 4. Salvataggio CSV
    # -------------------------------------------------------------------------
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(intestazioni)
        writer.writerows(righe)

    print(f"\nRisultati salvati in '{output_csv}'")

