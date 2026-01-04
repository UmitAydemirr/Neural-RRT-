import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import qmc
import random
from model import NeuralPlannerGAT
from torch_geometric.data import Data

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pt"
OUTPUT_DIR = Path(__file__).parent / "test_results"
K_NEIGHBORS = 25
NODE_DENSITY_FACTOR = 64


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
        if obs[0] == 'daire':
            d = np.hypot(px - obs[1], py - obs[2]) - obs[3]
        else:
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


def generate_random_map(width=None, height=None):
    if width is None:
        width = random.randint(100, 500)
    if height is None:
        height = random.randint(100, 500)
    
    obstacles = []
    area = width * height
    num_obstacles = max(5, int(area / 2000))
    
    for _ in range(num_obstacles):
        if random.random() < 0.5:
            r = random.randint(3, 15)
            x = random.randint(r, width - r)
            y = random.randint(r, height - r)
            obstacles.append(('daire', x, y, r))
        else:
            w = random.randint(5, 20)
            h = random.randint(5, 20)
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            obstacles.append(('dikdortgen', x, y, w, h))
    
    for _ in range(1000):
        sx = random.uniform(10, width - 10)
        sy = random.uniform(10, height - 10)
        if not is_collision([sx, sy], obstacles, padding=5):
            start = np.array([sx, sy])
            break
    else:
        return None
    
    for _ in range(1000):
        gx = random.uniform(10, width - 10)
        gy = random.uniform(10, height - 10)
        dist = np.hypot(gx - start[0], gy - start[1])
        if not is_collision([gx, gy], obstacles, padding=5) and dist > 50:
            goal = np.array([gx, gy])
            break
    else:
        return None
    
    return {
        'width': width,
        'height': height,
        'obstacles': obstacles,
        'start': start,
        'goal': goal
    }


def create_graph_from_map(map_data):
    width, height = map_data['width'], map_data['height']
    start, goal = map_data['start'], map_data['goal']
    obstacles = map_data['obstacles']
    
    target_nodes = int((width * height) / NODE_DENSITY_FACTOR)
    target_nodes = max(500, min(target_nodes, 4000))
    
    valid_nodes = [start, goal]
    sampler = qmc.Halton(d=2, scramble=True)
    
    while len(valid_nodes) < target_nodes:
        samples = sampler.random(n=int((target_nodes - len(valid_nodes)) * 1.5)) * [width, height]
        for pos in samples:
            if len(valid_nodes) >= target_nodes:
                break
            if not is_collision(pos, obstacles, padding=2):
                valid_nodes.append(pos)
    
    nodes = np.array(valid_nodes)
    diag = np.hypot(width, height)
    
    features = []
    sdf_values = []
    for px, py in nodes:
        sdf = get_sdf(px, py, obstacles)
        sdf_values.append(sdf)
        features.append([
            px / width,
            py / height,
            np.linalg.norm([px, py] - start) / diag,
            np.linalg.norm([px, py] - goal) / diag,
            np.clip(sdf / 40.0, -1.0, 1.0)
        ])
    
    tree = cKDTree(nodes)
    _, indices = tree.query(nodes, k=K_NEIGHBORS + 1)
    edge_src, edge_dst, edge_validity = [], [], []
    
    for i in range(len(nodes)):
        for j in indices[i][1:]:
            is_valid = check_line_of_sight(nodes[i], nodes[j], obstacles)
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_validity.extend([float(is_valid), float(is_valid)])
    
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        sdf=torch.tensor(sdf_values, dtype=torch.float),
        edge_validity=torch.tensor(edge_validity, dtype=torch.float),
        raw_coords=torch.tensor(nodes, dtype=torch.float),
        map_size=torch.tensor([width, height], dtype=torch.float),
        start_idx=torch.tensor(0),
        goal_idx=torch.tensor(1)
    )
    return data, nodes


def run_inference(model, data, device, max_steps=100, max_visits=3):
    model.eval()
    data = data.to(device)
    
    start_idx = data.start_idx.item()
    goal_idx = data.goal_idx.item()
    
    path = [start_idx]
    current_node = start_idx
    visit_counts = torch.zeros(data.num_nodes, device=device)
    
    with torch.no_grad():
        for _ in range(max_steps):
            visit_counts[current_node] += 1
            
            edge_mask = data.edge_index[0] == current_node
            neighbors = data.edge_index[1, edge_mask]
            
            if len(neighbors) == 0:
                return None, path
            
            neighbor_edge_indices = torch.where(edge_mask)[0]
            neighbor_sdf = data.sdf[neighbors]
            neighbor_visits = visit_counts[neighbors]
            edge_validity = data.edge_validity[neighbor_edge_indices]
            
            all_scores = model(data.x, data.edge_index)
            predictions = all_scores[neighbors]
            
            safe_logits = predictions.clone()
            safe_logits[neighbor_visits >= max_visits] = -float('inf')
            safe_logits[neighbor_sdf < 0] = -float('inf')
            safe_logits[edge_validity < 0.5] = -float('inf')
            
            if torch.all(safe_logits == -float('inf')):
                return None, path
            
            best_idx = torch.argmax(safe_logits).item()
            current_node = neighbors[best_idx].item()
            path.append(current_node)
            
            if current_node == goal_idx:
                return path, path
    
    return None, path


def visualize_result(map_data, nodes, path, success, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    width, height = map_data['width'], map_data['height']
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    for obs in map_data['obstacles']:
        if obs[0] == 'daire':
            circle = Circle((obs[1], obs[2]), obs[3], color='gray', alpha=0.7)
            ax.add_patch(circle)
        else:
            rect = Rectangle((obs[1], obs[2]), obs[3], obs[4], color='gray', alpha=0.7)
            ax.add_patch(rect)
    
    ax.scatter(nodes[:, 0], nodes[:, 1], c='lightblue', s=3, alpha=0.5, label='Nodes')
    
    if path:
        path_coords = nodes[path]
        ax.plot(path_coords[:, 0], path_coords[:, 1], 'b-', linewidth=2, label='Model Path')
        ax.scatter(path_coords[:, 0], path_coords[:, 1], c='blue', s=20, zorder=5)
    
    ax.scatter(*map_data['start'], c='lime', s=200, marker='^', edgecolors='black', 
               linewidths=2, label='Start', zorder=10)
    ax.scatter(*map_data['goal'], c='red', s=200, marker='*', edgecolors='black', 
               linewidths=2, label='Goal', zorder=10)
    
    status = "SUCCESS" if success else "FAILED"
    color = "green" if success else "red"
    ax.set_title(f"Neural Path Planner - {status}", fontsize=16, color=color)
    ax.legend(loc='upper right')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Result saved: {save_path}")


def load_model(checkpoint_path, device):
    model = NeuralPlannerGAT(in_channels=5, hidden_channels=64, heads=4, dropout=0.2)
    
    if not checkpoint_path.exists():
        drive_path = Path("/content/drive/MyDrive/GAT_Checkpoints/best_model.pt")
        if drive_path.exists():
            checkpoint_path = drive_path
        else:
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs, best loss: {checkpoint['best_loss']:.4f}")
    
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = load_model(CHECKPOINT_PATH, device)
    if model is None:
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    num_tests = 10
    success_count = 0
    attempt = 0
    max_attempts = 100
    
    print(f"\nRunning {num_tests} tests...")
    
    while success_count < num_tests and attempt < max_attempts:
        attempt += 1
        print(f"\n--- Attempt {attempt} ---")
        
        map_data = generate_random_map()
        if map_data is None:
            print("Failed to generate valid map, retrying...")
            continue
        
        print(f"Map: {map_data['width']}x{map_data['height']}, {len(map_data['obstacles'])} obstacles")
        
        data, nodes = create_graph_from_map(map_data)
        print(f"Graph: {len(nodes)} nodes, {data.edge_index.shape[1]} edges")
        
        result_path, traversed_path = run_inference(model, data, device)
        
        if result_path is not None:
            success_count += 1
            print(f"SUCCESS! Path found with {len(result_path)} nodes")
            save_path = OUTPUT_DIR / f"success_{success_count}.png"
            visualize_result(map_data, nodes, result_path, True, save_path)
        else:
            print(f"FAILED. Traversed {len(traversed_path)} nodes before stuck")
            save_path = OUTPUT_DIR / f"failed_attempt_{attempt}.png"
            visualize_result(map_data, nodes, traversed_path, False, save_path)
    
    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{num_tests} successful")
    print(f"Total attempts: {attempt}")
    print(f"Success rate: {success_count/attempt*100:.1f}%")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
