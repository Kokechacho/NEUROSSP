import math
import graphgen as gg
import random
import time
import copy
import os
import csv
from functools import reduce
from math import gcd

def gcd_scaling_graph(H):
    """Scale all weights in graph H by GCD."""
    # Collect weights
    all_ws = [int(w) for u in H for v, w in H[u].items() if w < INF and u != v]
    if not all_ws:
        return H
    g = reduce(gcd, all_ws)
    # Apply scaling
    H_gcd = copy.deepcopy(H)
    for u in H_gcd:
        for v in H_gcd[u]:
            if H_gcd[u][v] < INF and u != v:
                H_gcd[u][v] = H_gcd[u][v] // g
    return H_gcd


def rank_scaling_graph(H):
    """
    Scale all weights in graph H by rank‐mapping.
    Each unique finite weight is assigned an integer rank starting at 1
    in ascending order of unique weights.
    """
    # Collect all finite, non‐self weights
    all_ws = [w for u in H for v, w in H[u].items() if w < INF and u != v]
    if not all_ws:
        return H
    
    # Build rank map
    uniq = sorted(set(all_ws))
    rank = {w: i+1 for i, w in enumerate(uniq)}
    
    # Apply mapping
    H_rank = copy.deepcopy(H)
    for u in H_rank:
        for v in H_rank[u]:
            w = H_rank[u][v]
            if w < INF and u != v:
                H_rank[u][v] = rank[w]
    return H_rank


def log_scaling_graph(H, base=10.0):
    """
    Scale all weights in graph H by an integer logarithmic transform.
    For each finite weight w > 0, computes:
        tick = ceil(log_base(w))
    and ensures tick >= 1 (so every edge has at least one tick).
    """
    H_log = copy.deepcopy(H)
    for u in H_log:
        for v in H_log[u]:
            w = H_log[u][v]
            if w < INF and u != v:
                if w <= 0:
                    # Puedes omitir, elevar, o dar error:
                    H_log[u][v] = 1  # O usar un valor por defecto mínimo
                else:
                    tick = math.ceil(math.log(w, base))
                    H_log[u][v] = max(1, tick)
    return H_log

def truncation_scaling_graph(H, divisor=10):
    """
    Scale all weights in graph H by truncating each positive weight to the nearest lower multiple of `divisor`.
    For each finite weight w > 0, computes:
        tick = floor(w / divisor) * divisor
    and ensures tick >= 1 (so every edge has at least one tick).
    """
    H_trunc = copy.deepcopy(H)
    for u in H_trunc:
        for v in H_trunc[u]:
            w = H_trunc[u][v]
            if w < INF and u != v:
                if w <= 0:
                    H_trunc[u][v] = 1  # Default minimum
                else:
                    tick = (w // divisor) * divisor
                    H_trunc[u][v] = max(1, tick)
    return H_trunc

def root_scaling_graph(H, alpha=2.0):
    """
    Compress weights using a root-based transform:
        tick(w) = ceil(w ** (1/alpha)) + 1
    """
    import copy, math
    H_scaled = copy.deepcopy(H)
    for u in H_scaled:
        for v in H_scaled[u]:
            w = H_scaled[u][v]
            if w < INF and u != v:
                if w <= 0:
                    H_scaled[u][v] = 1
                else:
                    tick = math.ceil(w ** (1 / alpha)) + 1
                    H_scaled[u][v] = max(1, tick)
    return H_scaled

def calcular_coste_camino(G, path):
    coste = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        coste += G[u][v]
    return abs(coste)

# Valor para representar conexiones inexistentes.
INF = float('inf')

# Semilla global para reproducibilidad
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)

# ----------------------------------
# Función de simulación
# ----------------------------------
def simular_camino(grafo, inicio, destino, timeout=None):
    """
    Simula la propagación de "bits" en un grafo dirigido y calcula:
      - camino: la lista de nodos desde 'inicio' hasta 'destino'
      - ticks: total de ciclos consumidos
      - avg_spikes: media percentual de neuronas que “spikean” por paso

    Parámetros:
      grafo: dict of dicts {u: {v: cost, ...}, ...}
      inicio: nodo fuente
      destino: nodo destino
      timeout: límite de pasos (si es None, se calcula automáticamente)

    Devuelve:
      (camino, ticks, avg_spikes)
      donde avg_spikes ∈ [0,1]
    """
    # Número total de neuronas en la red
    total_neurons = len(grafo)

    camino = []
    current_destino = destino
    ticks = 0

    # Calcular timeout si no se da
    if timeout is None:
        timeout = sum(
            coste for u in grafo for coste in grafo[u].values()
            if coste < INF
        ) + 1

    # Contadores para spikes y pasos
    total_spikes = 0
    total_steps = 0

    while current_destino != inicio:
        active = {
            vecino: [coste, inicio]
            for vecino, coste in grafo[inicio].items()
            if coste < INF
        }

        propagado = False
        pasos = 0

        while not propagado:
            pasos += 1
            total_steps += 1

            if pasos > timeout:
                # Calcular promedio percentual hasta el momento
                if total_steps > 0:
                    avg_per_step = total_spikes / total_steps
                    avg_spikes = avg_per_step / total_neurons
                else:
                    avg_spikes = 0
                print(f"Timeout alcanzado. No se pudo propagar hasta {current_destino}.")
                return [], total_steps, avg_spikes

            ready_nodes = []
            next_active = {}
            for nodo, (contador, remitente) in active.items():
                nuevo_contador = contador - 1
                if nuevo_contador <= 0:
                    ready_nodes.append((nodo, remitente))
                else:
                    next_active[nodo] = [nuevo_contador, remitente]

            # Acumular spikes de este paso
            total_spikes += len(ready_nodes)

            for nodo, remitente in ready_nodes:
                if nodo == current_destino:
                    propagado = True
                    nuevo_destino = remitente
                    ticks += pasos
                    break
                for vecino, coste in grafo[nodo].items():
                    if coste < INF:
                        nc = coste
                        if vecino not in next_active or nc < next_active[vecino][0]:
                            next_active[vecino] = [nc, nodo]

            if propagado:
                break
            active = next_active
            if not active:
                if total_steps > 0:
                    avg_per_step = total_spikes / total_steps
                    avg_spikes = avg_per_step / total_neurons
                else:
                    avg_spikes = 0
                print(f"No hay más nodos activos. No se pudo alcanzar {current_destino}.")
                return [], total_steps, avg_spikes

        camino.insert(0, current_destino)
        current_destino = nuevo_destino

    camino.insert(0, inicio)

    # Cálculo final de la media percentual de spikes
    if total_steps > 0:
        avg_per_step = total_spikes / total_steps
        avg_spikes = avg_per_step / total_neurons
    else:
        avg_spikes = 0

    return camino, total_steps, avg_spikes


# ----------------------------------
# Conjuntos de grafos de prueba
# ----------------------------------
graph_sets = gg.cargar_todos_los_graph_sets()

# ----------------------------------
# Simulación y medición de tiempo
# ----------------------------------

os.makedirs("C:/Users/alvar/Documents/NEURO/resultados", exist_ok=True)

# Ejecutar también el paper's algorithm
for set_name, graphs in graph_sets.items():
    print(f"--- Graph set: {set_name} ---")
    nombre_archivo = f"C:/Users/alvar/Documents/NEURO/resultados/resultado_{set_name}_nuestro_trun.csv"
    with open(nombre_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "set_name", "graph_id", "n", "E", "density", "mean_w", "disp",
            "inicio", "destino", "path", "path_len", "execution_time_ms", "cost", "energy", "ticks"
        ])

        for idx, G in enumerate(graphs, start=1):
            n = len(G)
            edges = sum(1 for u in G for w in G[u].values() if w < INF)
            density = edges / (n * (n - 1))

            weights = [w for u in G for w in G[u].values() if w < INF]
            mean_w = sum(weights) / len(weights)
            var_w = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            disp = math.sqrt(var_w)

            print(f"Graph #{idx}: n={n}, |E|={edges}, density={density:.3f}, mean_w={mean_w:.2f}, disp={disp:.2f}")

            inicio, destino = 0, n-1
            H = copy.deepcopy(G)

            # H2 = gcd_scaling_graph(H)
            # H2 = rank_scaling_graph(H)
            # H2 = log_scaling_graph(H)
            # H2 = truncation_scaling_graph(H)
            # H2 = embedding_scaling_graph(H)
            H2 = root_scaling_graph(H)

            timeout = sum(w for u in H2 for w in H2[u].values() if w < INF)+1

            t0 = time.perf_counter()
            path, ticks, m_spike = simular_camino(H2, inicio, destino, timeout)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000  # ms
            # elapsedd = ideal_neuromorphic_time(ticks)
            coste = calcular_coste_camino(H, path)


            writer.writerow([
                set_name,
                idx,
                n,
                edges,
                round(density, 5),
                round(mean_w, 5),
                round(disp, 5),
                inicio,
                destino,
                str(path),
                len(path),
                round(coste, 5),
                ticks
            ])
