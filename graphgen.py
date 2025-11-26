import random
from itertools import product
import csv
import os

INF = float('inf')

OUTPUT_DIR = "C:/Users/alvar/Documents/NEURO/grafos_generados"

def guardar_grafo_csv(graph, path):
    """
    Guarda un grafo en un archivo CSV con columnas: source,target,weight
    """
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'weight'])
        for u in graph:
            for v in graph[u]:
                w = graph[u][v]
                if w != INF and u != v:
                    writer.writerow([u, v, round(w, 3)])

def exportar_todos_los_grafos(graph_sets):
    """
    Guarda todos los grafos en carpetas individuales con nombres descriptivos
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for set_name, graphs in graph_sets.items():
        set_dir = os.path.join(OUTPUT_DIR, set_name)
        os.makedirs(set_dir, exist_ok=True)
        for i, graph in enumerate(graphs):
            file_path = os.path.join(set_dir, f"graph_{i}.csv")
            guardar_grafo_csv(graph, file_path)

def cargar_grafo_csv(path):
    """
    Carga un grafo desde un archivo CSV con columnas: source,target,weight.
    Devuelve un diccionario {source: {target: weight, ...}, ...}
    """
    graph = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row['source'])
            v = int(row['target'])
            w = float(row['weight'])
            if u not in graph:
                graph[u] = {}
            graph[u][v] = w
    # Asegurar que todos los nodos están definidos, incluso sin salidas
    max_node = max(graph.keys(), default=-1)
    for node in range(max_node + 1):
        if node not in graph:
            graph[node] = {}
    return graph

def cargar_todos_los_graph_sets(directorio=OUTPUT_DIR):
    """
    Carga todos los conjuntos de grafos desde una estructura de carpetas.
    Devuelve un diccionario: {set_name: [grafo1, grafo2, ...], ...}
    """
    graph_sets = {}
    for set_name in os.listdir(directorio):
        set_path = os.path.join(directorio, set_name)
        if os.path.isdir(set_path):
            graphs = []
            for archivo in sorted(os.listdir(set_path)):
                if archivo.endswith(".csv"):
                    path_csv = os.path.join(set_path, archivo)
                    graph = cargar_grafo_csv(path_csv)
                    graphs.append(graph)
            graph_sets[set_name] = graphs
    return graph_sets

def generate_er_graph(n, p, weight_dist='uniform', weight_params=(1, 10), seed=None):
    if seed is not None:
        random.seed(seed)
    
    graph = {i: {} for i in range(n)}  # Asegura que todos los nodos están definidos
    
    for i in range(n):
        for j in range(n):
            if i == j:
                graph[i][j] = 0
            elif random.random() < p:
                if weight_dist == 'uniform':
                    low, high = weight_params
                    w = random.uniform(low, high)
                elif weight_dist == 'exponential':
                    scale, = weight_params
                    w = random.expovariate(1 / scale)
                else:
                    w = weight_params[0]
                if w < 1:
                    w = 1
                graph[i][j] = int(w)
    
    # Asegurar que cada nodo tiene al menos una conexión entrante o saliente
    all_nodes = set(graph.keys())
    connected_nodes = set(i for i in graph for j in graph[i] if graph[i][j] != INF and i != j)

    for node in all_nodes - connected_nodes:
        # Forzar una arista saliente si no tiene ninguna conexión
        target = random.choice([n for n in range(n) if n != node])
        if weight_dist == 'uniform':
            low, high = weight_params
            w = random.uniform(low, high)
        elif weight_dist == 'exponential':
            scale, = weight_params
            w = random.expovariate(1 / scale)
        else:
            w = weight_params[0]
        graph[node][target] = int(w)
    
    return graph

# 1. Definir los parámetros base para la malla
PARAM_GRID = {
    'sizes': [
        ('small', 20),
        ('medium', 50),
        ('large', 200),
        ('ultraLarge', 1000)
    ],
    'densities': [
        ('superSparse', 0.01),
        ('sparse', 0.05),
        ('medium', 0.25),
        ('dense', 0.5),
        ('superDense', 0.9)
    ],
    'weight_dists': [
        ('uniform', 'uniform', (1, 10)),
        ('exp', 'exponential', (5,)),
        ('const', 'constant', (1,)),
        ('highdelays', 'uniform', (100, 1000))
    ]
}

# 2. Filtrar combinaciones no deseadas (opcional)
def is_valid_combination(size, density, weight_dist):
    size_name, n = size
    dens_name, p = density
    dist_name, dist_type, params = weight_dist
    
    # Ejemplo: Evitar combinaciones muy pesadas computacionalmente
    if n <= 1 or p > 1:
        return False
    return True

# 3. Generador automático de configuraciones
def generate_graph_configs():
    configs = []
    for size, density, weight_dist in product(
        PARAM_GRID['sizes'],
        PARAM_GRID['densities'],
        PARAM_GRID['weight_dists']
    ):
        if is_valid_combination(size, density, weight_dist):
            size_name, n = size
            dens_name, p = density
            dist_name, dist_type, params = weight_dist
            
            config_name = f"{size_name}_{dens_name}_{dist_name}"
            configs.append((config_name, n, p, dist_type, params))
    
    return configs

# 4. Función para crear todos los graph_sets
def create_graph_sets(base_seed=42, graphs_per_config=5):
    configs = generate_graph_configs()
    graph_sets = {}
    
    for idx, config in enumerate(configs):
        name, n, p, dist, params = config
        graphs = []
        
        for i in range(graphs_per_config):
            seed = base_seed + (idx * 100) + i  # Semillas únicas
            graph = generate_er_graph(
                n=n,
                p=p,
                weight_dist=dist,
                weight_params=params,
                seed=seed
            )
            graphs.append(graph)
        
        graph_sets[name] = graphs
    
    return graph_sets

# 5. Ejecutar todo
if __name__ == "__main__":
    print("Generando grafos...")
    graph_sets = create_graph_sets()
    print("Guardando CSVs...")
    exportar_todos_los_grafos(graph_sets)
    print("✅ Proceso completado. Grafos guardados en:", OUTPUT_DIR)