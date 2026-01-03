import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from pathlib import Path
from model import NeuralPlannerGAT

DATA_DIR = Path(__file__).parent / "processed_tensors"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LOG_DIR = Path(__file__).parent / "runs"


class PathPlannerLoss(nn.Module):
    def __init__(self, obstacle_weight=5.0, revisit_weight=3.0, progress_weight=2.0,
                 terminal_weight=10.0, edge_weight=5.0, sdf_margin=2.0):
        super().__init__()
        self.obstacle_weight = obstacle_weight
        self.revisit_weight = revisit_weight
        self.progress_weight = progress_weight
        self.terminal_weight = terminal_weight
        self.edge_weight = edge_weight
        self.sdf_margin = sdf_margin
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predicted_logits, neighbor_coords, neighbor_sdf, target_pos,
                visit_counts, current_dist, is_terminal, goal_mask, edge_validity):
        
        predicted_probs = torch.softmax(predicted_logits, dim=0)
        
        dist_to_target = torch.norm(neighbor_coords - target_pos, dim=1)
        ideal_scores = 1.0 / (dist_to_target + 1e-5)
        target_probs = torch.softmax(ideal_scores, dim=0)
        base_loss = self.kl_loss(predicted_probs.log().unsqueeze(0), target_probs.unsqueeze(0))
        
        danger_level = torch.clamp(self.sdf_margin - neighbor_sdf, min=0.0)
        danger_score = torch.exp(danger_level) - 1.0
        obstacle_loss = torch.sum(predicted_probs * danger_score) * self.obstacle_weight
        
        revisit_loss = torch.sum(predicted_probs * (visit_counts ** 2)) * self.revisit_weight
        
        neighbor_dists = torch.norm(neighbor_coords - target_pos, dim=1)
        progress = current_dist - neighbor_dists
        normalized_progress = progress / (current_dist + 1e-5)
        progress_loss = -torch.sum(predicted_probs * normalized_progress) * self.progress_weight
        
        terminal_loss = torch.tensor(0.0, device=predicted_logits.device)
        if is_terminal:
            prob_on_goal = torch.sum(predicted_probs * goal_mask.float())
            terminal_loss = (1 - prob_on_goal) * self.terminal_weight
        
        invalid_edges = 1.0 - edge_validity
        edge_loss = torch.sum(predicted_probs * invalid_edges) * self.edge_weight
        
        total_loss = base_loss + obstacle_loss + revisit_loss + progress_loss + terminal_loss + edge_loss
        
        components = {
            'base': base_loss.item(),
            'obstacle': obstacle_loss.item(),
            'revisit': revisit_loss.item(),
            'progress': progress_loss.item(),
            'terminal': terminal_loss.item(),
            'edge': edge_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, components


class Trainer:
    def __init__(self, model, device, lr=0.001, max_visits=3, max_steps=50):
        self.model = model.to(device)
        self.device = device
        self.max_visits = max_visits
        self.max_steps = max_steps
        
        self.criterion = PathPlannerLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.writer = SummaryWriter(LOG_DIR)
        self.best_loss = float('inf')
        self.global_step = 0
        
        CHECKPOINT_DIR.mkdir(exist_ok=True)

    def train_episode(self, data):
        self.model.train()
        data = data.to(self.device)
        
        start_idx = data.start_idx.item() if hasattr(data, 'start_idx') else 0
        goal_idx = data.goal_idx.item() if hasattr(data, 'goal_idx') else 1
        
        if hasattr(data, 'raw_coords'):
            target_pos = data.raw_coords[goal_idx]
        else:
            target_pos = data.x[goal_idx, :2] * data.map_size
        
        visit_counts = torch.zeros(data.num_nodes, device=self.device)
        current_node = start_idx
        episode_loss = 0.0
        episode_components = {k: 0.0 for k in ['base', 'obstacle', 'revisit', 'progress', 'terminal', 'edge', 'total']}
        steps = 0
        success = False
        
        self.optimizer.zero_grad()
        
        for step in range(self.max_steps):
            visit_counts[current_node] += 1
            
            edge_mask = data.edge_index[0] == current_node
            neighbors = data.edge_index[1, edge_mask]
            
            if len(neighbors) == 0:
                break
            
            neighbor_edge_indices = torch.where(edge_mask)[0]
            
            if hasattr(data, 'raw_coords'):
                neighbor_coords = data.raw_coords[neighbors]
                current_pos = data.raw_coords[current_node]
            else:
                neighbor_coords = data.x[neighbors, :2] * data.map_size
                current_pos = data.x[current_node, :2] * data.map_size
            
            neighbor_sdf = data.sdf[neighbors] if hasattr(data, 'sdf') else data.x[neighbors, 4] * 40.0
            neighbor_visits = visit_counts[neighbors]
            edge_validity = data.edge_validity[neighbor_edge_indices] if hasattr(data, 'edge_validity') else torch.ones(len(neighbors), device=self.device)
            
            current_dist = torch.norm(current_pos - target_pos)
            goal_mask = neighbors == goal_idx
            is_terminal = goal_mask.any().item()
            
            all_scores = self.model(data.x, data.edge_index)
            predictions = all_scores[neighbors]
            
            loss, components = self.criterion(
                predictions, neighbor_coords, neighbor_sdf, target_pos,
                neighbor_visits, current_dist, is_terminal, goal_mask, edge_validity
            )
            
            loss.backward(retain_graph=True)
            episode_loss += loss.item()
            for k, v in components.items():
                episode_components[k] += v
            steps += 1
            
            with torch.no_grad():
                safe_logits = predictions.clone()
                safe_logits[neighbor_visits >= self.max_visits] = -float('inf')
                safe_logits[neighbor_sdf < 0] = -float('inf')
                safe_logits[edge_validity < 0.5] = -float('inf')
                
                if torch.all(safe_logits == -float('inf')):
                    break
                
                best_idx = torch.argmax(safe_logits).item()
                current_node = neighbors[best_idx].item()
            
            if current_node == goal_idx:
                success = True
                break
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return episode_loss / max(steps, 1), episode_components, success, steps

    def train_epoch(self, dataloader, epoch):
        total_loss = 0.0
        total_components = {k: 0.0 for k in ['base', 'obstacle', 'revisit', 'progress', 'terminal', 'edge', 'total']}
        successes = 0
        total_steps = 0
        
        for batch_idx, data in enumerate(dataloader):
            loss, components, success, steps = self.train_episode(data)
            
            total_loss += loss
            for k, v in components.items():
                total_components[k] += v
            successes += int(success)
            total_steps += steps
            
            self.global_step += 1
            
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss, self.global_step)
                for k, v in components.items():
                    self.writer.add_scalar(f'Components/{k}', v / max(steps, 1), self.global_step)
                self.writer.add_scalar('Train/Success', int(success), self.global_step)
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        success_rate = successes / num_batches
        
        self.writer.add_scalar('Epoch/AvgLoss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/SuccessRate', success_rate, epoch)
        self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.scheduler.step(avg_loss)
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_loss, avg_components, success_rate

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step
        }
        
        path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']
        return checkpoint['epoch']


def load_dataset(data_dir, limit=None):
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit:
        files = files[:limit]
    
    dataset = []
    for f in files:
        try:
            data = torch.load(f)
            dataset.append(data)
        except:
            continue
    return dataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = load_dataset(DATA_DIR, limit=5000)
    print(f"Loaded {len(dataset)} graphs")
    
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    model = NeuralPlannerGAT(in_channels=5, hidden_channels=64, heads=4, dropout=0.2)
    trainer = Trainer(model, device, lr=0.001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        avg_loss, components, success_rate = trainer.train_epoch(train_loader, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Success: {success_rate:.2%}")
        print(f"  base={components['base']:.3f}, obstacle={components['obstacle']:.3f}, "
              f"revisit={components['revisit']:.3f}, progress={components['progress']:.3f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)
    
    trainer.writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()