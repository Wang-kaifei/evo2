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
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, context, predictions, scale_features):
        batch_size = context.size(0)
        
        # 编码上下文
        context_emb = self.context_encoder(context)  # [batch_size, seq_len, hidden_size]
        
        # 编码预测结果
        pred_emb = self.prediction_encoder(predictions)  # [n_parscale, batch_size, seq_len, hidden_size]
        
        # 编码特征
        feature_emb = self.feature_encoder(scale_features)  # [n_parscale, hidden_size]
        feature_emb = feature_emb.unsqueeze(1).expand(-1, batch_size, -1)  # [n_parscale, batch_size, hidden_size]
        
        # 计算注意力
        context_emb = context_emb.unsqueeze(0).expand(self.n_parscale, -1, -1, -1)  # [n_parscale, batch_size, seq_len, hidden_size]
        attn_output, _ = self.attention(
            context_emb.view(-1, context_emb.size(2), context_emb.size(3)),
            pred_emb.view(-1, pred_emb.size(2), pred_emb.size(3)),
            pred_emb.view(-1, pred_emb.size(2), pred_emb.size(3))
        )
        attn_output = attn_output.view(self.n_parscale, batch_size, -1, context_emb.size(3))
        
        # 合并特征
        combined = torch.cat([
            attn_output.mean(dim=2),  # [n_parscale, batch_size, hidden_size]
            feature_emb,  # [n_parscale, batch_size, hidden_size]
            pred_emb.mean(dim=2)  # [n_parscale, batch_size, hidden_size]
        ], dim=-1)
        
        # 计算reward
        rewards = self.output_layer(combined).squeeze(-1)  # [n_parscale, batch_size]
        
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_data:
            context, predictions, scale_features, targets = batch
            
            # 前向传播
            rewards = model(context, predictions, scale_features)
            
            # 计算损失
            loss = criterion(rewards, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            pred = rewards.argmax(dim=0)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss/len(train_data):.4f}")
        print(f"Train Accuracy: {100*correct/total:.2f}%")
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_data:
                context, predictions, scale_features, targets = batch
                rewards = model(context, predictions, scale_features)
                loss = criterion(rewards, targets)
                
                val_loss += loss.item()
                pred = rewards.argmax(dim=0)
                val_correct += (pred == targets).sum().item()
                val_total += targets.size(0)
        
        print(f"Val Loss: {val_loss/len(val_data):.4f}")
        print(f"Val Accuracy: {100*val_correct/val_total:.2f}%")
        print("-" * 50) 