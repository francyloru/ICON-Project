import heapq
import datetime
import gestore_modelli
from datetime import date, timedelta
import json

with open("colture.json", "r", encoding="utf-8") as file:
    COLTURE = json.load(file)

# =============================================================================
# 2. CONFIGURAZIONE E CARICAMENTO DATI
# =============================================================================

# 
#COLTURE = {
#    'Zucche':   {'durata': 90,  't_ideal': 25}, #https://medium.com/@gardenlover/ideal-temperature-for-pumpkin-plants-f946be5bc75f
#    'Patate':   {'durata': 120, 't_ideal': 17}, # https://potatoinsights.com/best-climate-and-soil-conditions-for-potato-farming/#:~:text=16%E2%80%9321%C2%B0C%20during%20the%20day
#    'Pomodori': {'durata': 110,  't_ideal': 25}, # https://eos.com/blog/how-to-grow-tomatoes/
#    'Carote':   {'durata': 120,  't_ideal': 19} # https://en.wikipedia.org/wiki/Carrot#:~:text=Le%20carote%20vengono,e%2070%20%C2%B0F)
#}

# Dizionario globale: TEMPERATURE[citta] = [t_giorno_1, t_giorno_2, ... t_giorno_365]
TEMPERATURE = {}

def carica_dati_meteo(ANNO_TARGET, CITTA):
    print(f"\n-- Caricamento modelli predittivi per l'anno {ANNO_TARGET}:")
    
    for citta in CITTA:
        # print("Citta:",citta)
        dati_raw = gestore_modelli.predici_temperature_anno_citta(citta, ANNO_TARGET)
        # Convertiamo il dizionario {(mese,giorno): temp} in una lista piatta [t0, t1, t2...]
        # per accesso veloce O(1) tramite indice.
        lista_temp = []
        curr = date(ANNO_TARGET, 1, 1)
        while curr.year == ANNO_TARGET:
            key = (curr.month, curr.day)
            t = dati_raw.get(key, 0.0) # Gestione errori se manca un giorno
            lista_temp.append(t)
            curr += timedelta(days=1)
            
        TEMPERATURE[citta] = lista_temp
        print(f"  - Carcati i dati meteo predetti per la città {citta}({len(lista_temp)} giorni).")

# =============================================================================
# 3. PRE-CALCOLO COSTI (Lookup Table)
# =============================================================================

COSTI_PRECALCOLATI = {} # [pianta][citta][giorno_start] -> costo totale

def precalcola_costi(ANNO_TARGET, CITTA):
    # Crea una matrice di costi. Invece di calcolare l'energia durante la ricerca,
    # calcoliamo qui: "Se pianto X a Y il giorno Z, quanto spendo?"
    for pianta, info in COLTURE.items():
        COSTI_PRECALCOLATI[pianta] = {}
        durata = info['durata']
        t_ideal = info['t_ideal']
        
        for citta in CITTA:
            temps = TEMPERATURE[citta]
            giorni_totali = len(temps)
            costi_citta = []
            
            for start_day in range(giorni_totali):
                end_day = start_day + durata
                
                # Vincolo: non possiamo sforare l'anno
                if end_day > giorni_totali:
                    costi_citta.append(float('inf'))
                else:
                    # Calcolo energia: somma delle differenze termiche giornaliere
                    costo_tot = sum(abs(t_ideal - temps[d]) for d in range(start_day, end_day))
                    costi_citta.append(costo_tot)
            
            COSTI_PRECALCOLATI[pianta][citta] = costi_citta

# =============================================================================
# 4. ALGORITMO DI RICERCA A* (A-Star)
# =============================================================================

def get_date_string(day_index, ANNO_TARGET):
    # Converte indice 0-364 in 'DD Mese'
    d = date(ANNO_TARGET, 1, 1) + timedelta(days=day_index)
    return d.strftime("%d %B")

def calcola_euristica(piante_rimanenti, CITTA):
    # Stima ottimistica (Lower Bound): somma dei costi minimi assoluti 
    # per le piante rimaste, ignorando conflitti di serra.
    
    h = 0
    for p in piante_rimanenti:
        # Trova il costo minimo per questa pianta tra tutte le città e tutti i giorni possibili
        min_global = float('inf')
        for c in CITTA:
            min_local = min(COSTI_PRECALCOLATI[p][c])
            if min_local < min_global:
                min_global = min_local
        if min_global != float('inf'):
            h += min_global
    return h

def trova_miglior_start_date(pianta, citta, giorno_minimo):
    # Cerca il giorno con costo minore per 'pianta' nella 'citta',
    # ma SOLO dopo 'giorno_minimo' (quando la serra si libera).
    
    costi = COSTI_PRECALCOLATI[pianta][citta]
    limit = len(costi)
    
    best_day = -1
    min_cost = float('inf')
    
    # Scansioniamo dal giorno libero fino a fine anno
    for d in range(giorno_minimo, limit):
        if costi[d] < min_cost:
            min_cost = costi[d]
            best_day = d
            
    return best_day, min_cost

def run_a_star(ANNO_TARGET, CITTA):
    # Struttura nodo: (F_Score, G_Score, [Disponibilità Serre], [Piante Rimaste], [Log Azioni])
    # Disponibilità Serre è una tupla: (giorno_libero_Bari, giorno_libero_Lecce, ...)
    
    print("\n-- Avvio della ricerca con A*.")
    piante_init = tuple(sorted(COLTURE.keys()))
    disp_init = tuple([0] * len(CITTA))
    
    # Priority Queue
    start_h = calcola_euristica(piante_init, CITTA)
    start_node = (start_h, 0, disp_init, piante_init, [])
    
    open_set = [start_node]
    visited_states = set() # Per evitare cicli o percorsi ridondanti
    
    cnt = 0
    
    while open_set:
        cnt += 1
        f, g, disp, piante, storia = heapq.heappop(open_set)
        
        # GOAL STATE: Nessuna pianta rimasta
        if not piante:
            return g, storia
        
        # Pruning stati già visitati con costo peggiore/uguale
        state_sig = (disp, piante)
        if state_sig in visited_states:
            continue
        visited_states.add(state_sig)
        
        # Espansione: Prendiamo la prima pianta della lista (strategia deterministica per ridurre branching)
        pianta_target = piante[0]
        restanti = piante[1:]
        
        # Proviamo ad assegnarla a ciascuna città
        for i, citta in enumerate(CITTA):
            giorno_libero = disp[i]
            
            # Cerca il miglior slot temporale disponibile da ora in poi
            best_start, costo_energia = trova_miglior_start_date(pianta_target, citta, giorno_libero)
            
            if best_start != -1:
                # Nuovi parametri
                new_g = g + costo_energia
                new_h = calcola_euristica(restanti, CITTA)
                new_f = new_g + new_h
                
                # Aggiorna disponibilità serra
                durata = COLTURE[pianta_target]['durata']
                new_disp = list(disp)
                new_disp[i] = best_start + durata # La serra è occupata fino alla fine
                
                new_action = {
                    'citta': citta,
                    'pianta': pianta_target,
                    'start': best_start,
                    'end': best_start + durata,
                    'costo': costo_energia
                }
                
                heapq.heappush(open_set, (new_f, new_g, tuple(new_disp), restanti, storia + [new_action]))

    print("\n-- Ricerca con A* terminata.")      
    return None, None


def cerca_soluzione(ANNO_TARGET, CITTA):
    # 1. Carica previsioni ML
    carica_dati_meteo(ANNO_TARGET, CITTA)
    
    # 2. Precalcola costi energetici per ogni combinazione
    precalcola_costi(ANNO_TARGET, CITTA)
    
    # 3. Esegui A*
    
    energia_tot, piano = run_a_star(ANNO_TARGET, CITTA)
    
    if piano:
        print("\n=== PIANO OTTIMALE TROVATO ===")
        print(f"  - Anno di riferimento: {ANNO_TARGET}")
        print(f"  - Energia Totale Stimata: {energia_tot:.1f} unità termiche\n")
        
        # Ordiniamo per data cronologica
        piano.sort(key=lambda x: x['start'])
        
        print(" " * 25, "Riepilogo:")
        print("-" * 75)
        print(f"|  {'COLTURA':<10} |    {'CITTÀ':<8}|           {'PERIODO OTTIMALE':<26} | {'COSTO':<6}|")
        print("-" * 75)
        
        for p in sorted(piano, key=lambda x: x['citta']):
            d_start = get_date_string(p['start'], ANNO_TARGET)
            d_end = get_date_string(p['end'], ANNO_TARGET)
            print(f"|  {p['pianta']:<10} |  {p['citta']:<9} | {d_start:<16} -> {d_end:<16} |{p['costo']:6.1f} |")
        print("-" * 75)
    else:
        print("Nessuna soluzione trovata (forse troppe colture per le serre disponibili).")

    