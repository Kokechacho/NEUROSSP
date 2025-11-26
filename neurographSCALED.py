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

def entropy_scaling_graph(H):
    """
    Scale all weights in graph H using entropy coding (Huffman coding).
    Weights are compressed based on frequency — more common weights use fewer bits.
    Returns a graph with the original structure, but weights replaced by the Huffman code lengths.
    Each weight w becomes: tick = len(Huffman(w)), which approximates -log₂(P(w))

    Ensures tick >= 1 for all edges.
    """
    import copy
    from collections import Counter
    import heapq

    # Step 1: Collect all finite, positive weights
    all_weights = []
    for u in H:
        for v in H[u]:
            w = H[u][v]
            if w < INF and u != v and w > 0:
                all_weights.append(int(w))

    # Step 2: Count frequency of each weight
    freq = Counter(all_weights)

    # Step 3: Build Huffman Tree
    heap = [[weight, [sym, ""]] for sym, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huff_map = {sym: code for sym, code in heap[0][1:]}

    # Step 4: Encode weights as code lengths
    H_entropy = copy.deepcopy(H)
    for u in H_entropy:
        for v in H_entropy[u]:
            w = H_entropy[u][v]
            if w < INF and u != v:
                code_len = len(huff_map.get(int(w), '1111111'))  # Default long code if missing
                H_entropy[u][v] = max(1, code_len)  # At least 1 tick per edge
            else:
                H_entropy[u][v] = INF

    return H_entropy



def estimate_energy(ticks: int,
                    num_neurons: int,
                    num_synapses: int,
                    alpha: float = 0.3
                   ) -> float:
    """
    Estimate total energy consumption (in pJ) over a given number of simulation ticks
    for a memristive spiking neuromorphic hardware model, omitting STDP events.

    Parameters
    ----------
    ticks : int
        Number of simulated clock cycles (β).
    num_neurons : int
        Total number of neurons (|V|).
    num_synapses : int
        Total number of synapses (|E|).
    alpha : float, optional
        Fraction of neurons firing each cycle (default: 0.3).

    Returns
    -------
    float
        Estimated total energy consumed (in picojoules).
    """

    # Per‐event energies (pJ) from Schuman et al. (2019)
    E_fire            = 12.50   # energy per neuron firing
    E_acc             = 1.45    # energy per synaptic accumulation
    E_idle_neuron     = 7.20    # energy per neuron idle
    E_idle_synapse    = 0.07    # energy per synapse idle

    # Number of each event type
    N_fire            = alpha * ticks * num_neurons
    N_acc             = alpha * ticks * num_synapses
    N_idle_neuron = max(0, ticks * num_neurons - N_fire)
    N_idle_synapse = max(0, ticks * num_synapses - N_acc)

    # Total energy
    E_total = (
        N_fire * E_fire +
        N_acc  * E_acc +
        N_idle_neuron  * E_idle_neuron +
        N_idle_synapse * E_idle_synapse
    )

    return E_total


def neuromorphic_fpga_estimate(
    ticks: int,
    n_nodes: int,
    activity_factor: float = 0.3,
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

    # 7) Return time in milliseconds
    return total_ns / 1e6

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
            # H2 = entropy_scaling_graph(H)
            H2 = root_scaling_graph(H)

            timeout = sum(w for u in H2 for w in H2[u].values() if w < INF)+1

            t0 = time.perf_counter()
            path, ticks, m_spike = simular_camino(H2, inicio, destino, timeout)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000  # ms
            elapsedd = neuromorphic_fpga_estimate(ticks=ticks,n_nodes=n, activity_factor=m_spike)
            energy = estimate_energy(ticks, n, edges, alpha=m_spike)
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
                round(elapsedd, 5),
                round(coste, 5),
                energy,
                ticks
            ])
