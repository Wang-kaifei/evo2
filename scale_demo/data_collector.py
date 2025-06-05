import torch
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class DataCollector:
    """收集训练数据的类"""
    def __init__(self, save_dir: str = "data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
    
    def collect(self, context: torch.Tensor, predictions: torch.Tensor, 
                scale_features: torch.Tensor, target: torch.Tensor):
        """收集训练数据
        
        Args:
            context: 输入序列 [batch_size, seq_len]
            predictions: 所有scale的预测 [n_parscale, batch_size, seq_len]
            scale_features: scale特征 [n_parscale, 2]
            target: 目标序列 [batch_size, seq_len-1]
        """
        batch_size = context.size(0)
        seq_len = context.size(1)
        
        # 确保target有正确的维度
        if target.dim() == 1:
            target = target.unsqueeze(0)  # [seq_len-1] -> [1, seq_len-1]
        
        # 计算每个流的预测是否正确
        is_correct = (predictions == target.unsqueeze(0)).float()  # [n_parscale, batch_size, seq_len-1]
        
        # 对每个位置，找到预测正确的流的索引
        correct_mask = is_correct > 0  # [n_parscale, batch_size, seq_len-1]
        
        # 检查每个位置是否所有scale预测都相同
        all_same = (predictions == predictions[0:1]).all(dim=0)  # [batch_size, seq_len-1]
        
        # 对每个位置，找出所有预测正确的scale
        correct_scales = torch.nonzero(correct_mask, as_tuple=True)[0]  # [n_correct]
        batch_indices = torch.nonzero(correct_mask, as_tuple=True)[1]  # [n_correct]
        pos_indices = torch.nonzero(correct_mask, as_tuple=True)[2]  # [n_correct]
        
        # 初始化correct_indices为-1（表示没有正确的预测或所有预测相同）
        correct_indices = torch.full((batch_size, seq_len-1), -1, device=context.device)
        
        # 对每个位置，选择一个scale
        for b in range(batch_size):
            for t in range(seq_len-1):
                # 如果所有scale预测都相同，保持-1
                if all_same[b, t]:
                    correct_indices[b, t] = -1
                    continue
                    
                # 找出当前位置所有正确的scale
                pos_mask = (batch_indices == b) & (pos_indices == t)
                if pos_mask.any():
                    # 如果有正确的预测，从正确的scale中随机选择一个
                    correct_scales_pos = correct_scales[pos_mask]
                    selected_scale = correct_scales_pos[torch.randint(0, len(correct_scales_pos), (1,), device=context.device)]
                    correct_indices[b, t] = selected_scale
                else:
                    # 如果没有正确的预测，保持-1
                    correct_indices[b, t] = -1
        
        # 保存数据
        for b in range(batch_size):
            sample = {
                'context': context[b].cpu().numpy().tolist(),
                'predictions': predictions[:, b].cpu().numpy().tolist(),
                'scale_features': scale_features.float().cpu().numpy().tolist(),
                'target': target[b].cpu().numpy().tolist(),
                'correct_scale': correct_indices[b].cpu().numpy().tolist()
            }
            self.data.append(sample)
    
    def save(self, filename: str = "training_data.json"):
        """保存收集的数据
        
        Args:
            filename: 保存的文件名
        """
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.data, f)
        print(f"Data saved to {save_path}")
    
    def load(self, filename: str = "training_data.json") -> List[Dict[str, Any]]:
        """加载保存的数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            加载的数据列表
        """
        load_path = self.save_dir / filename
        with open(load_path, 'r') as f:
            self.data = json.load(f)
        return self.data

class DataLoader:
    """数据加载器"""
    def __init__(self, data: List[Dict[str, Any]], batch_size: int = 32, max_seq_len: int = 1024):
        self.data = data
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_samples = len(data)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        """将序列填充到指定长度
        
        Args:
            seq: 输入序列
            max_len: 目标长度
            
        Returns:
            填充后的序列
        """
        if len(seq) > max_len:
            return seq[:max_len]
        return seq + [0] * (max_len - len(seq))
    
    def __iter__(self):
        """迭代器"""
        indices = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.data[idx] for idx in batch_indices]
            
            # 获取当前批次中最长序列的长度
            max_len = min(max(len(d['context']) for d in batch_data), self.max_seq_len)
            
            # 准备批次数据，并进行填充
            context = torch.tensor([self.pad_sequence(d['context'], max_len) for d in batch_data]).to('cuda:0')
            
            # 处理预测结果
            n_parscale = len(batch_data[0]['predictions'])  # 获取并行scale数量
            padded_predictions = []
            for d in batch_data:
                # 对每个样本的每个scale的预测进行填充
                sample_predictions = [self.pad_sequence(pred, max_len-1) for pred in d['predictions']]
                padded_predictions.append(sample_predictions)
            # 转换为tensor并调整维度顺序
            predictions = torch.tensor(padded_predictions).transpose(0, 1).to('cuda:0')  # [n_parscale, batch_size, seq_len-1]
            
            scale_features = torch.tensor([d['scale_features'] for d in batch_data]).to('cuda:0')
            correct_scales = torch.tensor([self.pad_sequence(d['correct_scale'], max_len-1) for d in batch_data]).to('cuda:0')
            
            yield context, predictions, scale_features, correct_scales
    
    def __len__(self):
        """返回批次数量"""
        return self.n_batches 