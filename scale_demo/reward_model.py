import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Reward Model for selecting the best prediction
    
    输入：
    - context: 当前token的上下文 [batch_size, seq_len]
    - predictions: 所有scale的预测结果 [n_parscale, batch_size, seq_len-1]
    - scale_features: 每个流的特征 [n_parscale, 2]
    - prediction_scores: 每个流的预测分数 [n_parscale, batch_size, seq_len-1]
    
    输出：
    - rewards: 每个预测token的得分 [n_parscale, batch_size, seq_len-1]
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
        
        # 分数编码器
        self.score_encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 输出层 - 预测每个token的得分
        self.token_output = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # 3个特征拼接
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 将模型转换为bfloat16
        self.to(dtype=torch.bfloat16)
    
    def forward(self, context: torch.Tensor, predictions: torch.Tensor, 
                scale_features: torch.Tensor, prediction_scores: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            context: 输入序列 [batch_size, seq_len]
            predictions: 所有scale的预测 [n_parscale, batch_size, seq_len-1]
            scale_features: scale特征 [n_parscale, 2]
            prediction_scores: 预测分数 [n_parscale, batch_size, seq_len-1]
            
        Returns:
            torch.Tensor: 每个预测token的reward分数 [n_parscale, batch_size, seq_len-1]
        """
        batch_size = context.size(0)
        seq_len = context.size(1)
        n_parscale = predictions.size(0)
        
        # 确保输入类型正确
        # Embedding层需要整数类型输入
        context = context.long()  # [batch_size, seq_len]
        predictions = predictions.long()  # [n_parscale, batch_size, seq_len-1]
        
        # 其他特征使用bfloat16
        scale_features = scale_features.to(dtype=torch.bfloat16)  # [n_parscale, 2]
        prediction_scores = prediction_scores.to(dtype=torch.bfloat16)  # [n_parscale, batch_size, seq_len-1]
        
        # 编码上下文
        context_emb = self.context_encoder(context)  # [batch_size, seq_len, hidden_size]
        context_emb = context_emb[:, :-1]  # [batch_size, seq_len-1, hidden_size]
        context_emb = context_emb.unsqueeze(0).expand(n_parscale, -1, -1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 编码预测
        pred_emb = self.prediction_encoder(predictions.reshape(-1, seq_len-1))  # [n_parscale * batch_size, seq_len-1, hidden_size]
        pred_emb = pred_emb.reshape(n_parscale, batch_size, seq_len-1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 编码scale特征
        flat_features = scale_features.reshape(-1, 2)  # [n_parscale * batch_size, 2]
        flat_emb = self.feature_encoder(flat_features)  # [n_parscale * batch_size, hidden_size]
        feature_emb = flat_emb.reshape(n_parscale, batch_size, -1)  # [n_parscale, batch_size, hidden_size]
        feature_emb = feature_emb.unsqueeze(2).expand(-1, -1, seq_len-1, -1)  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 编码预测分数
        score_emb = self.score_encoder(prediction_scores.unsqueeze(-1))  # [n_parscale, batch_size, seq_len-1, hidden_size]
        
        # 注意力机制
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
        combined = torch.cat([attn_output, feature_emb, score_emb], dim=-1)  # [n_parscale, batch_size, seq_len-1, hidden_size*3]
        
        # 计算每个预测token的reward分数
        rewards = self.token_output(combined)  # [n_parscale, batch_size, seq_len-1, 1]
        rewards = rewards.squeeze(-1)  # [n_parscale, batch_size, seq_len-1]
        
        return rewards

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 使用AdamW优化器
    criterion = nn.MSELoss()  # 使用MSE损失，因为prediction_scores是连续值
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in train_data:
            context, predictions, scale_features, prediction_scores = batch
            
            # 前向传播
            rewards = model(context, predictions, scale_features, prediction_scores)
            
            # 计算损失
            loss = criterion(rewards, prediction_scores)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        
        # 验证
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad(): # validate不计算梯度
            for batch in val_data:
                context, predictions, scale_features, prediction_scores = batch
                rewards = model(context, predictions, scale_features, prediction_scores)
                loss = criterion(rewards, prediction_scores)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        
        avg_val_loss = val_loss / val_samples
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)
    
    # 训练完成后进行最终评估
    print("\nFinal Evaluation:")
    reward_accuracy, original_accuracy = evaluate_reward_model(model, val_data)
    print(f"Training completed. Final reward model accuracy: {reward_accuracy:.2f}%")
    print(f"Improvement over original model: {reward_accuracy - original_accuracy:.2f}%")

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

def evaluate_reward_model(model, val_data):
    """评估reward model在验证集上的性能
    
    Args:
        model: reward model
        val_data: 验证数据
        
    Returns:
        tuple: (reward_model_accuracy, original_model_accuracy)
            - reward_model_accuracy: reward model的token预测准确率
            - original_model_accuracy: 原始模型的token预测准确率
    """
    model.eval()
    token_correct = 0
    token_total = 0
    original_correct = 0
    original_total = 0
    
    with torch.no_grad():
        for batch in val_data:
            context, predictions, scale_features, prediction_scores = batch
            batch_size = context.size(0)
            seq_len = context.size(1)
            n_parscale = predictions.size(0)
            # batch_size, seq_len = predictions.shape[1], predictions.shape[2] + 1  # 因为predictions没有对第一个token做预测，所以这里要加1
            
            # 1. 评估原始模型（使用第一个scale的预测）
            original_predictions = predictions[0, :, :]  # [batch_size, seq_len-1]
            target_tokens = context[:, 1:]  # [batch_size, seq_len-1]
            mask = (target_tokens != 0) & (original_predictions != 0)
            original_correct += ((original_predictions == target_tokens) & mask).sum().item()
            original_total += mask.sum().item()
            
            # 2. 评估reward model
            # 获取每个位置的reward分数
            rewards = model(context, predictions, scale_features, prediction_scores)  # [n_parscale, batch_size, seq_len-1]
            
            # 对每个位置，选择reward分数最高的预测
            selected_predictions = torch.zeros(batch_size, seq_len-1, dtype=torch.long, device=predictions.device)
            for b in range(batch_size):
                for t in range(seq_len-1):
                    # 选择reward分数最高的预测
                    best_scale = rewards[:, b, t].argmax().item()
                    selected_predictions[b, t] = predictions[best_scale, b, t]
            
            # 计算预测的token与真实token的匹配率
            token_correct += ((selected_predictions == target_tokens) & mask).sum().item()
            token_total += mask.sum().item()
    
    reward_model_accuracy = 100 * token_correct / token_total if token_total > 0 else 0
    original_model_accuracy = 100 * original_correct / original_total if original_total > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"Original Model Accuracy: {original_model_accuracy:.2f}%")
    print(f"Reward Model Accuracy: {reward_model_accuracy:.2f}%")
    print(f"Improvement: {reward_model_accuracy - original_model_accuracy:.2f}%")
    
    return reward_model_accuracy, original_model_accuracy 