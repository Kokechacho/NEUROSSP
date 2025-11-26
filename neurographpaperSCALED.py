import math
import copy
import graphgen as gg
import random
import time
import csv
import os
from functools import reduce
from math import gcd

# Valor para representar conexiones inexistentes.
INF = float('inf')

# Semilla global para reproducibilidad
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)

def calcular_coste_camino(G, path):
    coste = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        coste += G[u][v]
    return abs(coste)

def simular_camino_paper(grafo, inicio, destino, timeout=None):
    """
    Versión que:
      - Marca sinapsis usadas negativizando su coste en grafo (como antes).
      - Lleva `fired` como un dict { nodo: bool } en lugar de un set.
      - Backtracing comparando orig vs grafo.
      - Calcula avg_spikes: media percentual de neuronas que spikean por paso
        durante la fase de propagación.

    Devuelve:
      (path, pasos, avg_spikes)
      - path: lista de nodos desde inicio hasta destino (o [] si falla)
      - pasos: número de iteraciones en la fase de wavefront propagation (o -1 si falla)
      - avg_spikes: porcentaje medio de neuronas que disparan por paso (0..1)
    """
    orig = copy.deepcopy(grafo)
    total_neurons = len(grafo)

    # Inicializamos estado de disparo
    fired = { nodo: False for nodo in grafo }
    fired[inicio] = True

    # Calcular timeout si no se da
    if timeout is None:
        timeout = sum(
            coste for u in grafo for coste in grafo[u].values()
            if coste < INF
        ) + 1

    # Parámetros para la fase de propagación
    pasos = 0
    total_spikes = 0
    wave_steps = 0

    # --- 1) Wavefront propagation ---
    active = {
        v: [coste, inicio]
        for v, coste in grafo[inicio].items()
        if coste < INF
    }
    propagado = False

    while not propagado:
        pasos += 1
        wave_steps += 1

        if pasos > timeout:
            print(f"Timeout tras {timeout} pasos. No se alcanzó {destino}.")
            # Calculamos avg_spikes aunque falle
            avg_spikes = (total_spikes / wave_steps) / total_neurons if wave_steps else 0
            return [], -1, avg_spikes

        next_active = {}
        ready = []

        # Reducir retardos y filtrar fired
        for nodo, (cnt, remit) in active.items():
            if fired[nodo]:
                continue
            cnt -= 1
            if cnt <= 0:
                ready.append((nodo, remit))
            else:
                next_active[nodo] = [cnt, remit]

        # Contabilizar spikes de este paso
        total_spikes += len(ready)

        # Procesar ready
        for nodo, remit in ready:
            if fired[nodo]:
                continue
            fired[nodo] = True
            grafo[remit][nodo] = -orig[remit][nodo]
            if nodo == destino:
                propagado = True
                break
            for v2, coste2 in grafo[nodo].items():
                if coste2 > 0 and not fired[v2]:
                    prev = next_active.get(v2)
                    if prev is None or coste2 < prev[0]:
                        next_active[v2] = [coste2, nodo]

        if not propagado and not next_active:
            print(f"No hay más nodos activos. No se pudo alcanzar {destino}.")
            avg_spikes = (total_spikes / wave_steps) / total_neurons if wave_steps else 0
            return [], -1, avg_spikes

        active = next_active

    # --- 2) Back-tracing comparando orig vs grafo ---
    path = [destino]
    current = destino
    hops_left = total_neurons

    while current != inicio and hops_left > 0:
        encontrado = False
        for u in grafo:
            if orig[u].get(current, INF) > 0 and grafo[u].get(current, 0) < 0:
                path.insert(0, u)
                current = u
                encontrado = True
                break
        if not encontrado:
            print("Backtracing fallido: no se encontró arista marcada.")
            avg_spikes = (total_spikes / wave_steps) / total_neurons if wave_steps else 0
            return [], -1, avg_spikes
        hops_left -= 1

    if current != inicio:
        print("Backtracing abortado por límite de hops.")
        avg_spikes = (total_spikes / wave_steps) / total_neurons if wave_steps else 0
        return [], -1, avg_spikes

    # Cálculo final de la media percentual de spikes
    avg_spikes = (total_spikes / wave_steps) / total_neurons if wave_steps else 0
    return path, pasos, avg_spikes

graph_sets = gg.cargar_todos_los_graph_sets()

os.makedirs("C:/Users/alvar/Documents/NEURO/resultados", exist_ok=True)

# Ejecutar también el paper's algorithm
for set_name, graphs in graph_sets.items():
    print(f"--- Graph set: {set_name} ---")
    nombre_archivo = f"C:/Users/alvar/Documents/NEURO/resultados/resultado_{set_name}_paper.csv"
    with open(nombre_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "set_name", "graph_id", "n", "E", "density", "mean_w", "disp",
            "inicio", "destino", "path", "path_len", "execution_time_ms", "cost"
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

            # 1) Haz una copia profunda de G para no tocar el original
            H_scaled = copy.deepcopy(G)

            # 2) Recolecta todos los pesos finitos (excluyendo bucles)
            all_ws = [
                int(w)
                for u in H_scaled
                for v, w in H_scaled[u].items()
                if w < INF and u != v
            ]

            if all_ws:
                # 3) Calcula el gcd de todos los pesos
                g = reduce(gcd, all_ws)

                # 4) Divide cada peso por el gcd
                for u in H_scaled:
                    for v in H_scaled[u]:
                        w = H_scaled[u][v]
                        if w < INF and u != v:
                            # w_scaled es entero
                            H_scaled[u][v] = w // g

            timeout = sum(w for u in H for w in H[u].values() if w < INF)+1

            t0 = time.perf_counter()
            path, ticks, m_spike = simular_camino_paper(H_scaled, inicio, destino, timeout)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000  # ms

            coste = calcular_coste_camino(H, path)
            print(coste)

            print(f"  Removed path: {path}")
            print(f"  Execution time: {elapsed:.2f} ms\n")

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
                round(coste, 5)
            ])
