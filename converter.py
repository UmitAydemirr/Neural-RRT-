import torch
import numpy as np
import json
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from scipy.stats import qmc
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

BASE_DIR = Path(__file__).parent
JSON_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "processed_tensors"
DEBUG_DIR = BASE_DIR / "debug_images"

K_NEIGHBORS = 25
NODE_DENSITY_FACTOR = 64
SOFT_LABEL_SIGMA = 3.0

def is_collision(pos, obstacles, padding=0):
    px, py = pos
    for obs in obstacles:
        if obs[0] == 'daire':
            ox, oy, r = obs[1], obs[2], obs[3]
            if np.hypot(px - ox, py - oy) <= r + padding:
                return True
        elif obs[0] == 'dikdortgen':
            ox, oy, w, h = obs[1], obs[2], obs[3], obs[4]
            if (ox - padding <= px <= ox + w + padding) and (oy - padding <= py <= oy + h + padding):
                return True
    return False

def get_sdf(px, py, obstacles):
    min_dist = float('inf')
    for obs in obstacles:
        obs_type = obs[0]
        d = float('inf')
        if obs_type == 'daire':
            ox, oy, r = obs[1], obs[2], obs[3]
            d = np.hypot(px - ox, py - oy) - r
        elif obs_type == 'dikdortgen':
            ox, oy, w, h = obs[1], obs[2], obs[3], obs[4]
            dx = max(ox - px, 0, px - (ox + w))
            dy = max(oy - py, 0, py - (oy + h))
            if (ox < px < ox + w) and (oy < py < oy + h):
                d = -min(px - ox, ox + w - px, py - oy, oy + h - py)
            else:
                d = np.hypot(dx, dy)
        if d < min_dist:
            min_dist = d
    return min_dist

def check_line_of_sight(p1, p2, obstacles, num_samples=10):
    for t in np.linspace(0, 1, num_samples):
        point = p1 + t * (p2 - p1)
        if is_collision(point, obstacles):
            return False
    return True

def save_debug_image(idx, width, height, obstacles, nodes, labels, path_nodes, start, goal):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    for obs in obstacles:
        if obs[0] == 'daire':
            circle = Circle((obs[1], obs[2]), obs[3], color='black', alpha=0.5)
            ax.add_patch(circle)
        elif obs[0] == 'dikdortgen':
            rect = Rectangle((obs[1], obs[2]), obs[3], obs[4], color='black', alpha=0.5)
            ax.add_patch(rect)
    
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=labels, cmap='viridis', s=5, alpha=0.8)
    plt.colorbar(sc, label='Path Probability')
    ax.plot(path_nodes[:, 0], path_nodes[:, 1], 'r-', linewidth=1, label='RRT* Path', alpha=0.6)
    ax.scatter(*start, c='lime', s=100, edgecolors='black', label='Start', marker='^')
    ax.scatter(*goal, c='red', s=100, edgecolors='black', label='Goal', marker='*')
    
    ax.legend(loc='upper right')
    ax.set_title(f"Map {idx} - Nodes: {len(nodes)}")
    
    save_path = DEBUG_DIR / f"map_debug_{idx}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def process_single_map(fpath, idx, save_debug=False):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        width = data['harita_boyut']['genislik']
        height = data['harita_boyut']['yukseklik']
        start = np.array(data['baslangic'])
        goal = np.array(data['hedef'])
        obstacles = data['engeller']
        if 'rrt_path' not in data or not data['rrt_path']:
            return False
        path_nodes = np.array(data['rrt_path'])
        
        target_nodes = int((width * height) / NODE_DENSITY_FACTOR)
        target_nodes = max(500, min(target_nodes, 4000))
        valid_nodes = [start, goal]
        sampler = qmc.Halton(d=2, scramble=True)
        
        while len(valid_nodes) < target_nodes:
            needed = target_nodes - len(valid_nodes)
            batch_size = int(needed * 1.5)
            samples = sampler.random(n=batch_size) * [width, height]
            for pos in samples:
                if len(valid_nodes) >= target_nodes:
                    break
                if not is_collision(pos, obstacles, padding=2):
                    valid_nodes.append(pos)
        nodes = np.array(valid_nodes)
        
        features = []
        diag = np.hypot(width, height)
        sdf_values = []

        for px, py in nodes:
            norm_pos = [px / width, py / height]
            d_start = np.linalg.norm([px, py] - start) / diag
            d_goal = np.linalg.norm([px, py] - goal) / diag
            sdf = get_sdf(px, py, obstacles)
            sdf_values.append(sdf)
            norm_sdf = np.clip(sdf / 40.0, -1.0, 1.0)
            features.append(norm_pos + [d_start, d_goal, norm_sdf])
            
        x = torch.tensor(features, dtype=torch.float)
        sdf_tensor = torch.tensor(sdf_values, dtype=torch.float)
        raw_coords = torch.tensor(nodes, dtype=torch.float)
        map_size = torch.tensor([width, height], dtype=torch.float)
        
        path_tree = cKDTree(path_nodes)
        dist_to_path, _ = path_tree.query(nodes, k=1)
        y_soft = np.exp(-(dist_to_path ** 2) / (2 * SOFT_LABEL_SIGMA ** 2))
        
        final_labels = []
        for i, val in enumerate(y_soft):
            if sdf_values[i] < 2.0:
                final_labels.append(0.0)
            else:
                final_labels.append(val if val > 0.01 else 0.0)
        
        y = torch.tensor(final_labels, dtype=torch.float).unsqueeze(1)
        
        if save_debug:
            save_debug_image(idx, width, height, obstacles, nodes, final_labels, path_nodes, start, goal)

        tree = cKDTree(nodes)
        dists, indices = tree.query(nodes, k=K_NEIGHBORS + 1)
        edge_src, edge_dst = [], []
        edge_validity = []
        
        for i in range(len(nodes)):
            for neighbor_idx in indices[i][1:]:
                is_valid = check_line_of_sight(nodes[i], nodes[neighbor_idx], obstacles)
                edge_src.append(i)
                edge_dst.append(neighbor_idx)
                edge_validity.append(float(is_valid))
                edge_src.append(neighbor_idx)
                edge_dst.append(i)
                edge_validity.append(float(is_valid))
        
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_validity_tensor = torch.tensor(edge_validity, dtype=torch.float)
        
        data_obj = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            sdf=sdf_tensor,
            edge_validity=edge_validity_tensor,
            raw_coords=raw_coords,
            map_size=map_size,
            start_idx=torch.tensor(0),
            goal_idx=torch.tensor(1)
        )
        torch.save(data_obj, OUTPUT_DIR / f"data_{idx}.pt")
        return True
        
    except Exception as e:
        return False


def process_and_save():
    json_files = list(JSON_DIR.glob("*.json"))
    if len(json_files) == 0:
        alt = JSON_DIR / "json_maps"
        json_files = list(alt.glob("*.json"))
    
    if len(json_files) == 0:
        print("ERROR: No JSON files found.")
        return

    print(f"Processing {len(json_files)} files...")
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True)
    
    num_cores = multiprocessing.cpu_count()
    used_cores = max(1, num_cores - 1)
    print(f"Using {used_cores} CPU cores...")
    
    tasks = [(fpath, idx, idx % 500 == 0) for idx, fpath in enumerate(json_files)]
    
    results = Parallel(n_jobs=used_cores, verbose=10)(
        delayed(process_single_map)(fpath, idx, save_debug) for fpath, idx, save_debug in tasks
    )
    
    success_count = sum(results)
    print(f"Successfully processed: {success_count}/{len(json_files)}")

if __name__ == "__main__":
    process_and_save()
    print("Processing completed. Check 'debug_images' folder.")