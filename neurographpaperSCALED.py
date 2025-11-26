import math
import copy
import graphgen as gg
import random
import time
import csv
import os
from functools import reduce
from math import gcd

def backtracing_with_bus_estimate(
    path_length: int,
    n_nodes: int,
    graph_density: float,
    clock_mhz: float = 100.0,
    parallelism: int = 64,
    bus_capacity_ev: int = 64,
    bus_latency_ns: float = 50.0,
    event_cycles_per_event: int = 7,
    init_sync_factor: float = 1.25
) -> float:
    """
    Estimates the time in ms for the backtracing phase in a neuromorphic system,
    considering memory access, bus communication, and computation over incoming synapses.
    
    Parameters:
    - path_length: number of nodes in the backtracked path
    - n_nodes: total number of nodes in the graph
    - graph_density: density ∈ [0, 1], used to estimate number of incoming synapses per node
    - clock_mhz, parallelism, bus_capacity_ev, bus_latency_ns: architectural assumptions

    Returns:
    - estimated backtracing time in ms
    """
    # 1) Sinapsis a revisar por nodo ≈ densidad × número total de nodos
    synapses_per_node = max(1, int(graph_density * n_nodes))
    total_syn_checks = path_length * synapses_per_node

    # 2) Bus cost to access synaptic states
    bus_cycles = (total_syn_checks + bus_capacity_ev - 1) // bus_capacity_ev
    bus_ns = bus_cycles * bus_latency_ns

    # 3) Compute cost: each check needs delay + spike_time + comparison
    compute_cycles = event_cycles_per_event * (total_syn_checks / parallelism)

    # 4) Clock cycle duration in ns
    clk_ns = 1e3 / clock_mhz

    # 5) Total time
    total_ns = (compute_cycles * clk_ns) 
    total_ns *= init_sync_factor  # Add sync penalty

    return total_ns / 1e6  # convert to ms

def neuromorphic_fpga_estimate(
    ticks: int,
    n_nodes: int,
    path_len: int,
    density: float,
    activity_factor: float = 0.05,
    clock_mhz: float = 100.0,
    parallelism: int = 64,
    bus_capacity_ev: int = 64,
    bus_latency_ns: float = 50.0,
    event_cycles_per_event: int = 7,  # ← basado en Caron et al. (2013)
    iface_overhead_ns: float = 300.0,
    init_sync_factor: float = 1.25
) -> float:
    """
    Estimates total execution time (ms) for 'ticks' timesteps in a neuromorphic FPGA architecture,
    assuming each spike is an atomic event.

    Parameters:
    - activity_factor × n_nodes = events per tick
    - bus_capacity_ev = events that can be transmitted per bus cycle
    - bus_latency_ns = latency per bus cycle (includes arbitration/routing)
    - event_cycles_per_event = clock cycles to process one event (memory + routing + sync)
        [Caron et al. (2013) reports ~7 cycles/event]
    - parallelism = number of processing elements (PEs)
    - clock_mhz = system clock in MHz
    - iface_overhead_ns, init_sync_factor = global overhead components

    Returns:
    - total estimated time in milliseconds
    """

    # 1) Number of events per tick
    ev_per_tick = max(1, int(n_nodes * activity_factor))

    # 2) Number of bus cycles needed per tick
    bus_cycles = (ev_per_tick + bus_capacity_ev - 1) // bus_capacity_ev
    bus_ns = bus_cycles * bus_latency_ns

    # 3) Total compute cycles per tick
    comp_cycles = event_cycles_per_event * (ev_per_tick / parallelism)

    # 4) Clock cycle duration in ns
    clk_ns = 1e3 / clock_mhz

    # 5) Total time per tick (ns)
    tick_ns = bus_ns + comp_cycles * clk_ns

    # 6) Total time over all ticks (ns), including interface overhead and sync factor
    total_ns = ticks * tick_ns + iface_overhead_ns
    total_ns *= init_sync_factor

    bt= backtracing_with_bus_estimate(path_len, n_nodes, density)

    # 7) Return time in milliseconds
    return total_ns / 1e6 + bt

def ideal_neuromorphic_time(ticks: int,
                            update_rate_hz: float = 10e6) -> float:
    """
    Convierte un número de ticks lógicos en un chip neuromórfico
    ideal (sin contenciones ni overhead) a tiempo real en milisegundos.

    Args:
        ticks: número de pasos de reloj lógicos.
        update_rate_hz: tasa de actualización de neuronas por núcleo (Hz).
                         Por defecto 10e6 (Loihi).

    Returns:
        Tiempo en milisegundos (float).
    """
    # 1) Duración de un tick en segundos
    tick_s = 1.0 / update_rate_hz
    # 2) Tiempo total en segundos
    total_s = ticks * tick_s
    # 3) Convertir a milisegundos
    return total_s * 1e3

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
            elapsedd = neuromorphic_fpga_estimate(ticks, n, len(path), density, activity_factor=m_spike)

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
                round(elapsedd, 5),
                round(coste, 5)
            ])
