import torch
import torch.nn.functional as F

class SimpleParScaleWrapper:
    """简化版的 ParScale 包装器
    
    通过对输入进行不同模式的扰动来模拟多尺度效果
    """
    def __init__(self, model, n_parscale=4, reward_model=None):
        self.model = model
        self.n_parscale = n_parscale  # 并行尺度的数量
        self.reward_model = reward_model  # 奖励模型
        
        # 获取模型的数据类型
        self.dtype = next(model.model.parameters()).dtype
        
        # 初始化扰动强度
        self.noise_scales = torch.nn.Parameter(
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                        device='cuda:0', dtype=self.dtype)
        )
        
        # 初始化丢弃率
        self.dropout_rates = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                        device='cuda:0', dtype=self.dtype)
    
    def apply_scale_specific_perturbation(self, logits, scale_idx, seq_len):
        """对每个尺度应用特定的扰动模式"""
        batch_size, seq_len, hidden_size = logits.shape
        
        if scale_idx == 0:  # 随机丢弃：完全随机
            mask = (torch.rand_like(logits) > self.dropout_rates[scale_idx]).bool()
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())
            
        elif scale_idx == 1:  # 随机丢弃：每隔一个位置
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, ::2, :] = False
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())
            
        elif scale_idx == 2:  # 随机丢弃：连续块
            block_size = seq_len // 4
            mask = torch.ones_like(logits, dtype=torch.bool)
            for d in range(hidden_size):
                start_idx = torch.randint(0, seq_len - block_size, (batch_size,), device='cuda:0')
                for b in range(batch_size):
                    mask[b, start_idx[b]:start_idx[b] + block_size, d] = False
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())
            
        elif scale_idx == 3:  # 局部扰动：只扰动序列的前1/3
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, :seq_len//3, :] = True
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 4:  # 局部扰动：只扰动序列的后1/3
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, -seq_len//3:, :] = True
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 5:  # 交替扰动：每隔一个位置扰动
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, ::2, :] = True
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 6:  # 交替扰动：每隔两个位置扰动
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, ::3, :] = True
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        else:  # 混合扰动：结合多种模式
            base_noise = torch.randn_like(logits) * self.noise_scales[scale_idx] * 0.5
            position = torch.arange(seq_len, device='cuda:0', dtype=self.dtype)
            position = position.view(1, -1, 1)
            frequencies = torch.linspace(0.05, 0.15, hidden_size, device='cuda:0', dtype=self.dtype)
            periodic_noise = torch.sin(position * frequencies.view(1, 1, -1)) * self.noise_scales[scale_idx] * 0.3
            position_weights = torch.linspace(0, 1, seq_len, device='cuda:0', dtype=self.dtype)
            position_weights = position_weights.view(1, -1, 1)
            dimension_weights = torch.linspace(0.5, 1.5, hidden_size, device='cuda:0', dtype=self.dtype)
            progressive_noise = torch.randn_like(logits) * self.noise_scales[scale_idx] * 0.2 * position_weights * dimension_weights.view(1, 1, -1)
            
            noise = base_noise + periodic_noise + progressive_noise
        
        return noise
    
    def forward(self, input_ids):
        """简化的 ParScale 前向传播
        
        Args:
            input_ids: 输入序列的token ids [batch_size, seq_len]
            
        Returns:
            tuple: (处理后的logits, combined_accuracy, all_predictions)
        """
        batch_size = input_ids.size(0)
        
        # 获取原始模型的输出
        with torch.inference_mode():
            original_output = self.model.model.forward(input_ids)
            if isinstance(original_output, tuple):
                if isinstance(original_output[0], tuple):
                    if isinstance(original_output[0][0], tuple):
                        logits = original_output[0][0][0]
                    else:
                        logits = original_output[0][0]
                else:
                    logits = original_output[0]
            else:
                logits = original_output
        
        # 存储每个尺度的输出
        scale_outputs = []
        seq_len = logits.size(1)
        
        # 对每个尺度进行处理
        for i in range(self.n_parscale):
            # 应用特定尺度的扰动
            noise = self.apply_scale_specific_perturbation(logits, i, seq_len)
            
            # 添加扰动后的输出
            scale_output = logits + noise
            scale_outputs.append(scale_output)
        
        # 计算每个尺度的loss
        scale_losses = []
        target_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
        
        # 存储每个尺度的预测结果
        all_predictions = []
        
        for i, output in enumerate(scale_outputs):
            # 计算交叉熵损失
            pred_logits = output[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            # 计算准确率
            pred_tokens = torch.argmax(pred_logits, dim=-1)  # [batch_size, seq_len-1]
            accuracy = (target_ids == pred_tokens).float().mean().item()
            
            scale_losses.append(loss)
            all_predictions.append(pred_tokens)
        
        # 计算组合准确率（只要有一个流预测正确就算正确）
        all_predictions = torch.stack(all_predictions)  # [n_parscale, batch_size, seq_len-1]
        correct_predictions = (all_predictions == target_ids.unsqueeze(0)).any(dim=0)  # [batch_size, seq_len-1]
        combined_accuracy = correct_predictions.float().mean().item()
        
        if self.reward_model is not None:
            # 使用奖励模型为每个位置选择最佳scale
            with torch.inference_mode():
                # 准备特征
                scale_features = torch.stack([
                    self.noise_scales,
                    self.dropout_rates
                ], dim=1)  # [n_parscale, 2]
                
                # 获取每个位置的reward
                rewards = self.reward_model(
                    input_ids,
                    all_predictions,
                    scale_features
                )  # [n_parscale, batch_size]
                
                # 选择每个位置reward最高的scale
                best_scale_indices = rewards.argmax(dim=0)  # [batch_size]
                
                # 根据最佳scale选择输出
                best_output = torch.stack([
                    scale_outputs[idx][b] for b, idx in enumerate(best_scale_indices)
                ])
        else:
            # 如果没有奖励模型，使用loss最小的scale
            best_scale_idx = torch.argmin(torch.stack(scale_losses))
            best_output = scale_outputs[best_scale_idx]
            
        return best_output, combined_accuracy, all_predictions 