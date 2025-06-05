import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Reward Model for selecting the best scale
    
    输入：
    - context: 当前token的上下文 [batch_size, seq_len]
    - predictions: 所有流的预测结果 [n_parscale, batch_size, seq_len]
    - scale_features: 每个流的特征 [n_parscale, n_features]
    
    输出：
    - rewards: 每个流的得分 [n_parscale, batch_size]
    """
    def __init__(self, vocab_size, hidden_size=768, n_parscale=8):
        super().__init__()
        self.n_parscale = n_parscale
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 预测编码器
        self.prediction_encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),  # noise_scale和dropout_rate
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 修改输入维度
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, context: torch.Tensor, predictions: torch.Tensor, scale_features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            context: 输入序列 [batch_size, seq_len]
            predictions: 所有scale的预测 [n_parscale, batch_size, seq_len-1]
            scale_features: scale特征 [n_parscale, batch_size, 2]
            
        Returns:
            rewards: 每个scale的reward分数 [n_parscale, batch_size, seq_len-1]
        """
        batch_size = context.size(0)
        seq_len = context.size(1)
        n_parscale = predictions.size(0)
        
        # 编码上下文
        context_emb = self.context_encoder(context)  # [batch_size, seq_len, hidden_size]
        
        # 编码预测
        pred_emb = self.prediction_encoder(predictions.reshape(-1, seq_len-1))  # [n_parscale * batch_size, seq_len-1, hidden_size]
        pred_emb = pred_emb.reshape(n_parscale, batch_size, seq_len-1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 编码scale特征
        flat_features = scale_features.reshape(-1, 2)  # [n_parscale * batch_size, 2]
        flat_emb = self.feature_encoder(flat_features)  # [n_parscale * batch_size, hidden_size]
        feature_emb = flat_emb.reshape(n_parscale, batch_size, -1)  # [n_parscale, batch_size, hidden_size]
        feature_emb = feature_emb.unsqueeze(2).expand(-1, -1, seq_len-1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 注意力机制
        context_emb = context_emb[:, :-1]  # [batch_size, seq_len-1, hidden_size]
        context_emb = context_emb.unsqueeze(0).expand(n_parscale, -1, -1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 准备注意力输入
        pred_emb_attn = pred_emb.reshape(-1, seq_len-1, pred_emb.size(-1))  # [n_parscale * batch_size, seq_len-1, hidden_size]
        context_emb_attn = context_emb.reshape(-1, seq_len-1, context_emb.size(-1))  # [n_parscale * batch_size, seq_len-1, hidden_size]
        
        # 计算注意力分数
        attn_output, _ = self.attention(
            query=pred_emb_attn,
            key=context_emb_attn,
            value=context_emb_attn
        )  # [n_parscale * batch_size, seq_len-1, hidden_size]
        
        # 恢复原始维度
        attn_output = attn_output.reshape(n_parscale, batch_size, seq_len-1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 合并特征
        combined = torch.cat([attn_output, feature_emb], dim=-1)  # [n_parscale, batch_size, seq_len-1, hidden_size*2]
        
        # 计算reward分数
        rewards = self.output_layer(combined)  # [n_parscale, batch_size, seq_len-1]
        
        return rewards

def evaluate_original_model(val_data):
    """评估原始模型在验证集上的性能
    
    Args:
        val_data: 验证数据
        
    Returns:
        float: token预测准确率
    """
    token_correct = 0
    token_total = 0
    
    with torch.no_grad():
        for batch in val_data:
            context, predictions, scale_features, targets = batch
            batch_size, seq_len = predictions.shape[1], predictions.shape[2]
            
            # 使用第一个scale的预测作为原始模型的预测
            original_predictions = predictions[0, :, :]  # [batch_size, seq_len - 1]
            # 使用context中的下一个token作为目标
            target_tokens = context[:, 1:]  # [batch_size, seq_len-1]
            
            # 创建mask，排除padding token (0)
            mask = (target_tokens != 0) & (original_predictions != 0)
            
            # 计算预测的token与真实token的匹配率，只考虑非padding的位置
            token_correct += ((original_predictions == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()
    
    accuracy = 100 * token_correct / token_total
    return accuracy

def train_reward_model(model, train_data, val_data, num_epochs=10, batch_size=32, learning_rate=1e-4):
    """训练reward model
    
    Args:
        model: RewardModel实例
        train_data: 训练数据
        val_data: 验证数据
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')  # 使用none以处理mask
    
    # 首先评估原始模型的性能
    original_accuracy = evaluate_original_model(val_data)
    print(f"\nOriginal Model Token Prediction Accuracy: {original_accuracy:.2f}%")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_data:
            context, predictions, scale_features, targets = batch
            
            # 前向传播
            rewards = model(context, predictions, scale_features)  # [n_parscale, batch_size, seq_len-1, 1]
            
            # 调整维度顺序，使其符合CrossEntropyLoss的要求
            rewards = rewards.squeeze(-1).transpose(0, 1)  # [batch_size, n_parscale, seq_len-1]
            
            # 创建mask，排除-1的位置
            mask = (targets != -1)
            
            # 临时替换-1为0，用于计算损失
            targets_for_loss = targets.clone()
            targets_for_loss[~mask] = 0
            
            # 计算损失，只考虑非-1的位置
            loss = criterion(rewards, targets_for_loss)
            loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)  # 添加平滑项
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率，只考虑非-1的位置
            pred = rewards.argmax(dim=1)  # [batch_size, seq_len-1]
            
            # 获取序列长度
            batch_size, seq_len = predictions.shape[1], predictions.shape[2] + 1
            
            # 创建mask，只排除-1的位置
            mask = (targets != -1)
            
            # 计算准确率，只考虑非-1的位置
            correct += ((pred == targets) & mask).sum().item()
            total += mask.sum().item()
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss/len(train_data):.4f}")
        print(f"Train Accuracy: {100*correct/total:.2f}%")
        
        # 验证
        model.eval()
        val_loss = 0
        val_scale_correct = 0
        val_scale_total = 0
        val_token_correct = 0
        val_token_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                context, predictions, scale_features, targets = batch
                batch_size, seq_len = predictions.shape[1], predictions.shape[2] + 1
                
                rewards = model(context, predictions, scale_features)
                rewards = rewards.squeeze(-1).transpose(0, 1)  # [batch_size, n_parscale, seq_len-1]
                
                # 创建mask，排除-1的位置
                mask = (targets != -1)
                
                # 临时替换-1为0，用于计算损失
                targets_for_loss = targets.clone()
                targets_for_loss[~mask] = 0
                
                # 计算损失，只考虑非-1的位置
                loss = criterion(rewards, targets_for_loss)
                loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)  # 添加平滑项
                
                val_loss += loss.item()
                
                # 计算scale选择准确率，只考虑非padding的位置
                for b in range(batch_size):
                    for t in range(seq_len-1):
                        # 如果predictions中所有scale在该位置都是0，说明这是padding位置
                        if (predictions[:, b, t] == 0).all() or targets[b, t].item() == -1:
                            continue
                        val_scale_total += 1
                        # 检查选择的scale是否与targets一致
                        selected_scale = rewards[b, :, t].argmax().item()  # 获取选择的scale索引
                        target_scale = targets[b, t].item()  # 获取目标scale索引
                        if selected_scale == target_scale:  # 比较两个整数
                            val_scale_correct += 1
                
                # 计算token预测准确率，考虑所有位置
                # 1. 获取每个位置选择的scale的预测
                selected_predictions = torch.zeros(batch_size, seq_len-1, dtype=torch.long, device=predictions.device)
                
                for b in range(batch_size):
                    for t in range(seq_len-1):
                        # 对所有位置都选择得分最高的scale
                        selected_scale = rewards[b, :, t].argmax().item()  # 转换为整数
                        selected_predictions[b, t] = predictions[selected_scale, b, t]
                
                # 2. 计算预测的token与真实token的匹配率
                # 使用context中的下一个token作为目标
                target_tokens = context[:, 1:]  # [batch_size, seq_len-1]

                
                # 创建mask，排除padding token (0)
                mask = (target_tokens != 0) & (selected_predictions != 0)

                
                # 计算预测的token与真实token的匹配率，只考虑非padding的位置
                val_token_correct += ((selected_predictions == target_tokens) & mask).sum().item()
                val_token_total += mask.sum().item()

                
            # 计算每个scale的准确率
            scale_accuracies = [0] * predictions.shape[0]  # 初始化每个scale的准确率
            scale_totals = [0] * predictions.shape[0]  # 初始化每个scale的总数
            
            with torch.no_grad():
                for val_batch in val_data:
                    val_context, val_predictions, val_scale_features, val_targets = val_batch
                    val_target_tokens = val_context[:, 1:]  # [batch_size, seq_len-1]
                    
                    # 创建mask，排除padding token (0)
                    val_mask = (val_target_tokens != 0)
                    
                    # 计算每个scale的准确率
                    for scale in range(predictions.shape[0]):
                        scale_predictions = val_predictions[scale]  # [batch_size, seq_len-1]
                        scale_correct = ((scale_predictions == val_target_tokens) & val_mask).sum().item()
                        scale_total = val_mask.sum().item()
                        scale_accuracies[scale] += scale_correct
                        scale_totals[scale] += scale_total
            
            print("\nScale-wise prediction accuracy:")
            for scale in range(predictions.shape[0]):
                accuracy = scale_accuracies[scale] / scale_totals[scale] if scale_totals[scale] > 0 else 0
                print(f"Scale {scale}: {accuracy:.2%}")
        
        print(f"Val Loss: {val_loss/len(val_data):.4f}")
        print(f"Val Scale Selection Accuracy: {100*val_scale_correct/val_scale_total:.2f}%")
        print(f"Val Token Prediction Accuracy: {100*val_token_correct/val_token_total:.2f}%")
        print("-" * 50)

def evaluate_reward_model(model, val_data):
    """评估reward model在验证集上的性能
    
    Args:
        model: reward model
        val_data: 验证数据
        
    Returns:
        float: scale选择准确率
        float: token预测准确率
    """
    model.eval()
    val_scale_correct = 0
    val_scale_total = 0
    val_token_correct = 0
    val_token_total = 0
    
    with torch.no_grad():
        for batch in val_data:
            context, predictions, scale_features, targets = batch
            
            # 1. 计算scale选择准确率
            pred = model(context, predictions, scale_features)  # [batch_size, seq_len-1, n_parscale]
            pred = pred.argmax(dim=-1)  # [batch_size, seq_len-1]
            
            # 计算每个位置预测的token与真实token的匹配率
            batch_size, seq_len = predictions.shape[1], predictions.shape[2]
            selected_predictions = torch.zeros(batch_size, seq_len-1, dtype=torch.long, device=predictions.device)
            for b in range(batch_size):
                for t in range(seq_len-1):
                    # 对所有位置都选择得分最高的scale
                    selected_scale = pred[b, t]
                    selected_predictions[b, t] = predictions[selected_scale, b, t]
            
            # 计算预测的token与真实token的匹配率
            target_tokens = context[:, 1:]  # [batch_size, seq_len-1]
            
            # 创建mask，排除padding token (0)
            mask = (target_tokens != 0) & (selected_predictions != 0)
            
            # 计算预测的token与真实token的匹配率，只考虑非padding的位置
            val_token_correct += ((selected_predictions == target_tokens) & mask).sum().item()
            val_token_total += mask.sum().item()
            
            # 计算scale选择准确率，只考虑非padding的位置
            for b in range(batch_size):
                for t in range(seq_len-1):
                    # 如果predictions中所有scale在该位置都是0，说明这是padding位置
                    if (predictions[:, b, t] == 0).all():
                        continue
                    val_scale_total += 1
                    # 检查选择的scale是否与targets一致
                    selected_scale = pred[b, t]
                    if selected_scale == targets[b, t]:
                        val_scale_correct += 1
    
    scale_accuracy = 100 * val_scale_correct / val_scale_total if val_scale_total > 0 else 0
    token_accuracy = 100 * val_token_correct / val_token_total if val_token_total > 0 else 0
    
    return scale_accuracy, token_accuracy 