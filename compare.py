"""
compare.py - Model vs RRT* Karşılaştırma
Aynı haritada Neural Model ve RRT* algoritmasını karşılaştırır
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
import math
from pathlib import Path
from scipy.spatial import cKDTree


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_colab():
    PROJECT_DIR = Path("/content/project/GAT/Neural-RRT--main")
else:
    PROJECT_DIR = Path(__file__).parent

import sys
sys.path.insert(0, str(PROJECT_DIR))
from model import NeuralPlannerGAT
from random_map_rrt import Node, Harita, BRRTStar


class ModelPathFinder:
    """Neural Model ile path bulma"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def find_path(self, graph_data, max_iterations=100, max_visits=3):
        """Model ile path bul, iterasyon sayısını döndür"""
        data = graph_data.to(self.device)
        raw_coords = data.raw_coords.cpu().numpy()
        start_idx = data.start_idx.item()
        goal_idx = data.goal_idx.item()
        
        path_indices = [start_idx]
        current_idx = start_idx
        visit_counts = torch.zeros(data.num_nodes, device=self.device)
        iterations = 0
        
        with torch.no_grad():
            for _ in range(max_iterations):
                iterations += 1
                visit_counts[current_idx] += 1
                
                # Komşuları bul
                edge_mask = data.edge_index[0] == current_idx
                neighbors = data.edge_index[1, edge_mask]
                
                if len(neighbors) == 0:
                    break
                
                neighbor_edge_indices = torch.where(edge_mask)[0]
                neighbor_sdf = data.sdf[neighbors]
                neighbor_visits = visit_counts[neighbors]
                edge_validity = data.edge_validity[neighbor_edge_indices]
                

                all_scores = self.model(data.x, data.edge_index)
                predictions = all_scores[neighbors]
                

                safe_logits = predictions.clone()
                safe_logits[neighbor_visits >= max_visits] = -float('inf')
                safe_logits[neighbor_sdf < 0] = -float('inf')
                safe_logits[edge_validity < 0.5] = -float('inf')
                
                if torch.all(safe_logits == -float('inf')):
                    break
                
                best_idx = torch.argmax(safe_logits).item()
                next_idx = neighbors[best_idx].item()
                path_indices.append(next_idx)
                current_idx = next_idx
                
                if current_idx == goal_idx:
                    break
        
        path_coords = [raw_coords[i].tolist() for i in path_indices]
        success = (current_idx == goal_idx)
        
        return {
            'path': path_coords,
            'path_indices': path_indices,
            'iterations': iterations,
            'success': success,
            'visited_count': len(set(path_indices))
        }


def create_random_map(width=100, height=100, num_obstacles=15):
    """Rastgele harita oluştur"""
    harita = Harita(width, height)
    

    for _ in range(num_obstacles):
        harita.rastgele_engel_ekle()
    

    padding = 10
    max_attempts = 100
    
    for _ in range(max_attempts):
        start_x = random.uniform(padding, width - padding)
        start_y = random.uniform(padding, height - padding)
        if not harita.carpisma_kontrol(start_x, start_y, padding=5.0):
            harita.baslangic = Node(start_x, start_y)
            break
    
    for _ in range(max_attempts):
        goal_x = random.uniform(padding, width - padding)
        goal_y = random.uniform(padding, height - padding)
        dist_to_start = math.hypot(goal_x - harita.baslangic.x, goal_y - harita.baslangic.y)
        if not harita.carpisma_kontrol(goal_x, goal_y, padding=5.0) and dist_to_start > 30:
            harita.hedef = Node(goal_x, goal_y)
            break
    
    return harita


def map_to_graph(harita, num_samples=300):
    """Haritayı PyTorch Geometric graph'a dönüştür"""
    from scipy.stats import qmc
    from scipy.ndimage import distance_transform_edt
    

    sampler = qmc.Halton(d=2, scramble=True)
    samples = sampler.random(n=num_samples)
    samples[:, 0] *= harita.width
    samples[:, 1] *= harita.height
    

    start_coord = [harita.baslangic.x, harita.baslangic.y]
    goal_coord = [harita.hedef.x, harita.hedef.y]
    

    valid_points = [start_coord, goal_coord]
    for p in samples:
        if not harita.carpisma_kontrol(p[0], p[1], padding=3.0):
            valid_points.append(p.tolist())
    
    coords = np.array(valid_points)
    n_nodes = len(coords)
    start_idx = 0
    goal_idx = 1
    

    k = min(15, n_nodes - 1)
    kdtree = cKDTree(coords)
    edge_list = []
    
    for i in range(n_nodes):
        _, indices = kdtree.query(coords[i], k=k+1)
        for j in indices[1:]:

            if not check_line_of_sight(harita, coords[i], coords[j]):
                edge_list.append([i, j])
                edge_list.append([j, i])
    

    edge_set = set()
    for e in edge_list:
        edge_set.add((e[0], e[1]))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    

    goal = coords[goal_idx]
    dist_to_goal = np.linalg.norm(coords - goal, axis=1)
    dist_to_goal_norm = dist_to_goal / (dist_to_goal.max() + 1e-6)

    resolution = 1.0
    grid_w = int(harita.width / resolution)
    grid_h = int(harita.height / resolution)
    obstacle_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    for obs in harita.engeller:
        if obs[0] == 'dikdortgen':
            _, x, y, w, h = obs
            x_min = max(0, int(x / resolution))
            x_max = min(grid_w, int((x + w) / resolution))
            y_min = max(0, int(y / resolution))
            y_max = min(grid_h, int((y + h) / resolution))
            obstacle_grid[y_min:y_max, x_min:x_max] = 1
        elif obs[0] == 'daire':
            _, cx, cy, r = obs
            for gy in range(grid_h):
                for gx in range(grid_w):
                    px, py = gx * resolution, gy * resolution
                    if (px - cx)**2 + (py - cy)**2 <= r**2:
                        obstacle_grid[gy, gx] = 1
    
    free_dist = distance_transform_edt(1 - obstacle_grid)
    
    sdf_values = []
    for coord in coords:
        gx = min(grid_w - 1, max(0, int(coord[0] / resolution)))
        gy = min(grid_h - 1, max(0, int(coord[1] / resolution)))
        sdf_values.append(free_dist[gy, gx])
    
    sdf_norm = np.array(sdf_values)
    sdf_norm = sdf_norm / (sdf_norm.max() + 1e-6)
    
    is_start = np.zeros(n_nodes)
    is_start[start_idx] = 1.0
    is_goal = np.zeros(n_nodes)
    is_goal[goal_idx] = 1.0
    
    x = np.column_stack([
        coords[:, 0] / harita.width,
        coords[:, 1] / harita.height,
        dist_to_goal_norm,
        sdf_norm,
        is_goal
    ])
    
    y = np.zeros(n_nodes)
    y[start_idx] = 1.0
    y[goal_idx] = 1.0
    
    edge_list = edge_index.t().numpy()
    edge_validity = []
    for e in edge_list:
        i, j = e[0], e[1]
        if not check_line_of_sight(harita, coords[i], coords[j]):
            edge_validity.append(1.0)
        else:
            edge_validity.append(0.0)
    
    from torch_geometric.data import Data
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float),
        raw_coords=torch.tensor(coords, dtype=torch.float),
        sdf=torch.tensor(sdf_norm, dtype=torch.float),
        edge_validity=torch.tensor(edge_validity, dtype=torch.float),
        start_idx=torch.tensor(start_idx),
        goal_idx=torch.tensor(goal_idx),
        map_size=torch.tensor([harita.width, harita.height])
    )
    
    return data


def check_line_of_sight(harita, p1, p2, step=0.5):
    """İki nokta arasında engel var mı kontrol et"""
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 0.1:
        return False
    
    steps = int(dist / step) + 1
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if harita.carpisma_kontrol(x, y, padding=2.0):
            return True
    return False


def calculate_path_length(path):
    """Path uzunluğunu hesapla"""
    if len(path) < 2:
        return 0
    
    total_length = 0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        total_length += math.hypot(dx, dy)
    return total_length


def visualize_comparison(harita, model_result, rrt_result, rrt_time_ms, model_time_ms, save_path=None):
    """İki sonucu yan yana görselleştir"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, result, title, time_ms, color in [
        (axes[0], model_result, "Neural Model (GAT)", model_time_ms, '#2ecc71'),
        (axes[1], rrt_result, "RRT* Algorithm", rrt_time_ms, '#e74c3c')
    ]:
        ax.set_xlim(0, harita.width)
        ax.set_ylim(0, harita.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        
        for obs in harita.engeller:
            if obs[0] == 'dikdortgen':
                _, x, y, w, h = obs
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1, edgecolor='#ff6b6b', facecolor='#2d2d44', alpha=0.8
                )
                ax.add_patch(rect)
            elif obs[0] == 'daire':
                _, cx, cy, r = obs
                circle = patches.Circle(
                    (cx, cy), r,
                    linewidth=1, edgecolor='#ff6b6b', facecolor='#2d2d44', alpha=0.8
                )
                ax.add_patch(circle)
        
        if result['success'] and result['path']:
            path = np.array(result['path'])
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=3, alpha=0.9, label='Path')
            

            ax.scatter(path[1:-1, 0], path[1:-1, 1], c=color, s=20, alpha=0.6, zorder=3)
        
        ax.scatter(harita.baslangic.x, harita.baslangic.y, c='#00ff00', s=200, 
                   marker='o', edgecolors='white', linewidths=2, zorder=5, label='Start')
        ax.scatter(harita.hedef.x, harita.hedef.y, c='#ff0000', s=200, 
                   marker='*', edgecolors='white', linewidths=2, zorder=5, label='Goal')
        
        path_length = calculate_path_length(result['path']) if result['path'] else 0
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        status_color = '#2ecc71' if result['success'] else '#e74c3c'
        
        info_text = f"""
{status}
Iterations: {result['iterations']}
Time: {time_ms:.1f} ms
Path Length: {path_length:.1f}
Nodes Visited: {result.get('visited_count', len(result['path']) if result['path'] else 0)}
"""
        
        props = dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', edgecolor=status_color, alpha=0.9)
        ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace', color='white', bbox=props)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    fig.patch.set_facecolor('#0d0d1a')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0d0d1a', edgecolor='none', bbox_inches='tight')
        print(f"📊 Karşılaştırma kaydedildi: {save_path}")
    
    plt.show()
    return fig


def run_comparison(model_path=None, num_comparisons=5, save_results=True):
    """Model ve RRT* karşılaştırması yap"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if model_path is None:
        if is_colab():
            model_path = "/content/drive/MyDrive/checkpoints/best_model.pt"
        else:
            model_path = PROJECT_DIR / "checkpoints" / "best_model.pt"
    
    print(f"📂 Model yükleniyor: {model_path}")
    
    model = NeuralPlannerGAT(in_channels=5, hidden_channels=64, heads=4, dropout=0.1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model yüklendi!")
    except Exception as e:
        print(f"⚠️  Model yüklenemedi: {e}")
        print("   Eğitilmemiş model ile devam ediliyor...")
    
    model.to(device)
    model.eval()
    
    model_finder = ModelPathFinder(model, device)
    
    results = {
        'model_wins': 0,
        'rrt_wins': 0,
        'ties': 0,
        'model_times': [],
        'rrt_times': [],
        'model_iterations': [],
        'rrt_iterations': [],
        'model_successes': 0,
        'rrt_successes': 0
    }
    
    print(f"\n{'='*60}")
    print(f"  🔬 MODEL vs RRT* KARŞILAŞTIRMASI ({num_comparisons} harita)")
    print(f"{'='*60}\n")
    
    for i in range(num_comparisons):
        print(f"\n{'─'*40}")
        print(f"  📍 Harita {i+1}/{num_comparisons}")
        print(f"{'─'*40}")
        
        harita = create_random_map(width=100, height=100, num_obstacles=random.randint(10, 20))
        
        print("  → Graph oluşturuluyor...")
        graph_data = map_to_graph(harita)
        print(f"    Nodes: {graph_data.x.size(0)}, Edges: {graph_data.edge_index.size(1)//2}")
        
        print("  → Neural Model çalışıyor...")
        start_time = time.time()
        model_result = model_finder.find_path(graph_data)
        model_time = (time.time() - start_time) * 1000
        
        print(f"    Model: {'✓' if model_result['success'] else '✗'} | "
              f"Iter: {model_result['iterations']} | Time: {model_time:.1f}ms")
        
        print("  → RRT* çalışıyor...")
        rrt_star = BRRTStar(harita, genisleme_mesafesi=4.0, max_iterasyon=5000, padding=5.0)
        
        start_time = time.time()
        rrt_path = rrt_star.planning()
        rrt_time = (time.time() - start_time) * 1000
        
        rrt_iterations = len(rrt_star.baslangic_tree) + len(rrt_star.end_tree)
        rrt_success = rrt_path is not None and len(rrt_path) > 0
        
        rrt_result = {
            'path': rrt_path if rrt_path else [],
            'iterations': rrt_iterations,
            'success': rrt_success,
            'visited_count': rrt_iterations
        }
        
        print(f"    RRT*:  {'✓' if rrt_success else '✗'} | "
              f"Iter: {rrt_iterations} | Time: {rrt_time:.1f}ms")
        
        # Sonuçları kaydet
        results['model_times'].append(model_time)
        results['rrt_times'].append(rrt_time)
        results['model_iterations'].append(model_result['iterations'])
        results['rrt_iterations'].append(rrt_iterations)
        
        if model_result['success']:
            results['model_successes'] += 1
        if rrt_success:
            results['rrt_successes'] += 1
        
        if model_result['success'] and not rrt_success:
            results['model_wins'] += 1
            winner = "🏆 MODEL"
        elif rrt_success and not model_result['success']:
            results['rrt_wins'] += 1
            winner = "🏆 RRT*"
        elif model_result['success'] and rrt_success:
            if model_time < rrt_time:
                results['model_wins'] += 1
                winner = "🏆 MODEL (faster)"
            elif rrt_time < model_time:
                results['rrt_wins'] += 1
                winner = "🏆 RRT* (faster)"
            else:
                results['ties'] += 1
                winner = "🤝 TIE"
        else:
            results['ties'] += 1
            winner = "❌ Both Failed"
        
        print(f"    Winner: {winner}")
        
        if save_results:
            save_path = PROJECT_DIR / f"comparison_{i+1}.png"
        else:
            save_path = None
        
        visualize_comparison(harita, model_result, rrt_result, rrt_time, model_time, save_path)
    
    print(f"\n{'='*60}")
    print("  📊 SONUÇ ÖZETİ")
    print(f"{'='*60}")
    print(f"""
  ┌─────────────────┬──────────────┬──────────────┐
  │                 │    MODEL     │     RRT*     │
  ├─────────────────┼──────────────┼──────────────┤
  │ Kazanma         │ {results['model_wins']:^12} │ {results['rrt_wins']:^12} │
  │ Başarı Oranı    │ {results['model_successes']/num_comparisons*100:^11.1f}% │ {results['rrt_successes']/num_comparisons*100:^11.1f}% │
  │ Ort. Süre (ms)  │ {np.mean(results['model_times']):^12.1f} │ {np.mean(results['rrt_times']):^12.1f} │
  │ Ort. İterasyon  │ {np.mean(results['model_iterations']):^12.1f} │ {np.mean(results['rrt_iterations']):^12.1f} │
  └─────────────────┴──────────────┴──────────────┘
    """)
    
    if results['ties'] > 0:
        print(f"  Berabere: {results['ties']}")
    
    speed_ratio = np.mean(results['rrt_times']) / np.mean(results['model_times'])
    print(f"\n  ⚡ Model, RRT*'dan ortalama {speed_ratio:.1f}x {'daha hızlı' if speed_ratio > 1 else 'daha yavaş'}!")
    
    return results


if __name__ == "__main__":
    results = run_comparison(num_comparisons=5, save_results=True)
