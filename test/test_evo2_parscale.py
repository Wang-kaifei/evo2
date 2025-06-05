import argparse
import csv
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from evo2 import Evo2

class ParscaleWrapper:
    """ParScale 模型包装器
    
    这个包装器为 Evo2 模型添加了 ParScale 功能。ParScale 的核心思想是：
    1. 通过多个并行的注意力尺度来增强模型的表达能力
    2. 每个尺度都有自己独立的 key 和 value 前缀
    3. 通过注意力平滑机制来优化不同尺度之间的交互
    """
    def __init__(self, model, n_parscale=4, n_tokens=48):
        self.model = model
        self.n_parscale = n_parscale  # 并行尺度的数量
        self.n_tokens = n_tokens      # 每个尺度的token数量
        
        # 获取模型的注意力头数和隐藏维度
        self.num_heads = model.model.config.num_attention_heads
        self.head_dim = model.model.config.hidden_size // self.num_heads
        
        # 获取模型的数据类型
        self.dtype = next(model.model.parameters()).dtype
        
        # 初始化每个尺度的前缀张量
        # prefix_k: 每个尺度的key前缀，形状为 [n_parscale, n_heads, n_tokens, head_dim]
        # prefix_v: 每个尺度的value前缀，形状为 [n_parscale, n_heads, n_tokens, head_dim]
        self.prefix_k = torch.nn.Parameter(torch.randn(n_parscale, self.num_heads, n_tokens, self.head_dim, 
                                                     device='cuda:0', dtype=self.dtype))
        self.prefix_v = torch.nn.Parameter(torch.randn(n_parscale, self.num_heads, n_tokens, self.head_dim, 
                                                     device='cuda:0', dtype=self.dtype))
        
        # 初始化尺度融合权重
        self.scale_weights = torch.nn.Parameter(torch.ones(n_parscale, device='cuda:0', dtype=self.dtype) / n_parscale)
        
        # 注意力平滑系数
        self.attn_smooth = 0.01
        
        # 初始化融合权重
        self.fusion_alpha = torch.nn.Parameter(torch.tensor(0.5, device='cuda:0', dtype=self.dtype))
        
        # 添加维度调整层，将输入维度调整为模型期望的隐藏层大小
        self.dim_adjust = torch.nn.Linear(512, model.model.config.hidden_size, 
                                        device='cuda:0', dtype=self.dtype)
        
        print(f"Debug - Model hidden size: {model.model.config.hidden_size}")
        print(f"Debug - Attention dimension: {self.num_heads * self.head_dim}")
        
    def get_input_embeddings(self, input_ids):
        """获取输入序列的嵌入向量
        
        Args:
            input_ids: 输入序列的token ids
            
        Returns:
            torch.Tensor: 输入序列的嵌入向量
        """
        # 使用模型的token嵌入层获取嵌入向量
        if hasattr(self.model.model, 'token_emb'):
            return self.model.model.token_emb(input_ids)
        elif hasattr(self.model.model, 'word_embeddings'):
            return self.model.model.word_embeddings(input_ids)
        else:
            # 如果没有找到嵌入层，使用默认的嵌入层
            return self.model.model.embed(input_ids)
        
    def compute_multi_scale_attention(self, input_ids, prefix_k, prefix_v):
        """计算多尺度注意力
        
        Args:
            input_ids: 输入序列的token ids
            prefix_k: 扩展后的key前缀
            prefix_v: 扩展后的value前缀
            
        Returns:
            torch.Tensor: 多尺度注意力输出
        """
        batch_size = input_ids.size(0)
        
        # 获取原始模型的输出作为基础嵌入
        with torch.inference_mode():
            original_output = self.model.model.forward(input_ids)
            if isinstance(original_output, tuple):
                if isinstance(original_output[0], tuple):
                    if isinstance(original_output[0][0], tuple):
                        base_embeddings = original_output[0][0][0]
                    else:
                        base_embeddings = original_output[0][0]
                else:
                    base_embeddings = original_output[0]
            else:
                base_embeddings = original_output
            
            print(f"Debug - Base embeddings shape: {base_embeddings.shape}")
        
        # 存储每个尺度的输出
        scale_outputs = []
        
        # 对每个尺度进行注意力计算
        for i in range(self.n_parscale):
            scale_k = prefix_k[i:i+1]  # [1, n_heads, n_tokens, head_dim]
            scale_v = prefix_v[i:i+1]  # [1, n_heads, n_tokens, head_dim]
            
            # 对每个注意力层进行处理
            current_x = base_embeddings
            for block in self.model.model.blocks:
                if hasattr(block, 'mixer'):
                    # 获取当前层的key和value
                    if hasattr(block.mixer, 'get_kv'):
                        k, v = block.mixer.get_kv(current_x)
                    else:
                        # 如果没有get_kv方法，使用默认的key和value计算
                        k = v = current_x
                    
                    # 确保维度匹配
                    if k.dim() == 3:  # [batch_size, seq_len, hidden_size]
                        k = k.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
                        v = v.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
                    
                    # 调整前缀维度以匹配
                    scale_k_expanded = scale_k.expand(batch_size, -1, -1, -1)  # [batch_size, n_heads, n_tokens, head_dim]
                    scale_v_expanded = scale_v.expand(batch_size, -1, -1, -1)  # [batch_size, n_heads, n_tokens, head_dim]
                    
                    # 计算注意力
                    if hasattr(block.mixer, 'compute_attention'):
                        # 在计算注意力之前，将前缀添加到query中
                        q = current_x.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
                        q = torch.cat([torch.zeros_like(scale_k_expanded, dtype=self.dtype), q], dim=2)  # 添加前缀
                        
                        # 创建注意力掩码
                        attention_mask = torch.ones((batch_size, 1, q.size(2), q.size(2)), 
                                                  device=q.device, dtype=self.dtype)
                        attention_mask[:, :, :self.n_tokens, :self.n_tokens] = 0  # 前缀部分可以互相注意
                        
                        # 计算注意力
                        attn_output = block.mixer.compute_attention(q, k, v, attention_mask)
                        
                        # 移除前缀部分
                        attn_output = attn_output[:, :, self.n_tokens:]
                    else:
                        # 对于非注意力层，直接使用原始输入
                        attn_output = current_x
                    
                    # 确保输出维度与输入一致
                    if attn_output.dim() == 4:  # [batch_size, n_heads, seq_len, head_dim]
                        attn_output = attn_output.reshape(batch_size, -1, attn_output.size(-1))
                    
                    current_x = attn_output
                else:
                    # 在进入block之前调整维度
                    if current_x.dim() == 2:  # [seq_len, hidden_size]
                        current_x = current_x.unsqueeze(0)  # [1, seq_len, hidden_size]
                    elif current_x.dim() == 3 and current_x.size(0) != batch_size:  # [seq_len, hidden_size, ?]
                        current_x = current_x.permute(1, 0, 2)  # [hidden_size, seq_len, ?]
                        current_x = current_x.reshape(batch_size, -1, current_x.size(-1))  # [batch_size, seq_len, hidden_size]
                    
                    
                    # 调整维度以匹配模型期望的隐藏层大小
                    if current_x.size(-1) != self.model.model.config.hidden_size:
                        current_x = self.dim_adjust(current_x)
                    
                    # 处理block的输出
                    block_output = block(current_x)
                    if isinstance(block_output, tuple):
                        current_x = block_output[0]  # 取第一个元素
                    else:
                        current_x = block_output
            
            scale_outputs.append(current_x)
        
        # 使用尺度权重融合不同尺度的输出
        weighted_output = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            weighted_output += self.scale_weights[i] * output
        
        return weighted_output
        
    def forward(self, input_ids):
        """ParScale 前向传播
        
        实现步骤：
        1. 将输入序列扩展到多个并行尺度
        2. 为每个尺度添加独立的前缀
        3. 计算多尺度注意力
        4. 应用注意力平滑机制
        5. 返回处理后的logits
        
        Args:
            input_ids: 输入序列的token ids
            
        Returns:
            tuple: (处理后的logits, None)
        """
        batch_size = input_ids.size(0)
        
        # 将前缀张量扩展到匹配批次大小
        prefix_k = repeat(self.prefix_k, 'n_parscale n_heads n_tokens head_dim -> (n_parscale b) n_heads n_tokens head_dim', 
                         b=batch_size // self.n_parscale)
        prefix_v = repeat(self.prefix_v, 'n_parscale n_heads n_tokens head_dim -> (n_parscale b) n_heads n_tokens head_dim', 
                         b=batch_size // self.n_parscale)
        
        # 计算多尺度注意力
        multi_scale_output = self.compute_multi_scale_attention(input_ids, prefix_k, prefix_v)
        
        # 获取原始模型的输出
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
        
        # 调整logits的维度以匹配multi_scale_output
        if logits.size(-1) != multi_scale_output.size(-1):
            logits = self.dim_adjust(logits)
        
        # 融合原始输出和多尺度输出
        # 使用注意力平滑机制来平衡两种输出
        if self.attn_smooth > 0:
            # 添加小噪声到logits以实现平滑
            noise = torch.randn_like(logits) * self.attn_smooth
            logits = logits + noise
        
        # 将多尺度输出与原始输出进行加权融合
        # 使用可学习的融合权重
        alpha = torch.sigmoid(self.fusion_alpha)  # 确保权重在[0,1]范围内
        logits = alpha * logits + (1 - alpha) * multi_scale_output
            
        return logits, None  # 返回元组以匹配原始模型接口

def read_prompts(input_file: Path) -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []
    
    try:
        with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            if not header:
                raise ValueError("Empty CSV file")
            
            # Read only first 10 rows after header (total 11 lines including header)
            for i, row in enumerate(reader):
                if i >= 10:  # Stop after reading 10 rows
                    break
                if row and len(row) > 0:  # Check if row is not empty
                    promptseqs.append(row[0])
        
        if not promptseqs:
            raise ValueError("No sequences found in the CSV file")
            
        return promptseqs
    except Exception as e:
        print(f"Error reading prompts file: {str(e)}")
        raise

def test_forward_pass_parscale(*, model, sequences, n_parscale=4, n_tokens=48):
    """测试ParScale模型的前向传播性能
    
    这个函数用于评估ParScale扩展后的模型性能，包括：
    1. 对每个输入序列进行前向传播
    2. 计算损失和准确率
    3. 输出详细的性能指标
    
    Args:
        model: 基础模型
        sequences: 测试序列列表
        n_parscale: 并行尺度数量
        n_tokens: 每个尺度的token数量
    """
    losses = []
    accuracies = []
    
    # 创建ParScale包装器
    parscale_model = ParscaleWrapper(model, n_parscale, n_tokens)
    
    for seq in sequences:
        # 将序列转换为模型输入格式
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to('cuda:0')
        print(f"Debug - Input shape: {input_ids.shape}")
        
        with torch.inference_mode():
            # 使用ParScale扩展进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
            print(f"Debug - Wrapper output type: {type(output)}")
            logits = output[0] if isinstance(output, tuple) else output
            print(f"Debug - Final logits shape: {logits.shape}")
            
            # 计算损失和准确率
            target_ids = input_ids[1:]  # 右移一位用于下一个token预测
            pred_logits = logits[0, :-1, :]
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                pred_logits, 
                target_ids.long()
            )
            
            # 获取预测结果
            pred_tokens = torch.argmax(pred_logits, dim=-1)
            
            # 计算准确率
            accuracy = (target_ids == pred_tokens).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
    
    # 打印序列结果
    print("\nSequence Results with ParScale:")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print("WARNING: Forward pass accuracy is below 50% on test sequence.")
    
    return accuracies, losses

def test_forward_pass_original(*, model, sequences):
    """测试原始模型的前向传播性能（不带ParScale扩展）
    
    这个函数用于评估原始模型的性能，作为ParScale扩展的基准对比。
    
    Args:
        model: 基础模型
        sequences: 测试序列列表
    """
    losses = []
    accuracies = []
    
    for seq in sequences:
        # 将序列转换为模型输入格式
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to('cuda:0')
        print(f"Debug - Original Input shape: {input_ids.shape}")
        
        with torch.inference_mode():
            # 原始模型的前向传播
            output = model(input_ids.unsqueeze(0))
            if isinstance(output, tuple):
                logits = output[0]
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                logits = output
                
            print(f"Debug - Original logits shape: {logits.shape}")
            
            # 计算损失和准确率
            target_ids = input_ids[1:]  # 右移一位用于下一个token预测
            pred_logits = logits[0, :-1, :]
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                pred_logits, 
                target_ids.long()
            )
            
            # 获取预测结果
            pred_tokens = torch.argmax(pred_logits, dim=-1)
            
            # 计算准确率
            accuracy = (target_ids == pred_tokens).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
    
    # 打印序列结果
    print("\nSequence Results (Original Model):")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print("WARNING: Forward pass accuracy is below 50% on test sequence.")
    
    return accuracies, losses

def compare_models(*, model, sequences, n_parscale=4, n_tokens=48):
    """比较原始模型和ParScale扩展的性能差异
    
    这个函数会：
    1. 测试原始模型的性能
    2. 测试ParScale模型的性能
    3. 输出详细的性能对比结果
    
    Args:
        model: 基础模型
        sequences: 测试序列列表
        n_parscale: 并行尺度数量
        n_tokens: 每个尺度的token数量
    """
    print("\n=== Testing Original Model ===")
    orig_accuracies, orig_losses = test_forward_pass_original(
        model=model,
        sequences=sequences
    )
    
    print("\n=== Testing ParScale Model ===")
    parscale_accuracies, parscale_losses = test_forward_pass_parscale(
        model=model,
        sequences=sequences,
        n_parscale=n_parscale,
        n_tokens=n_tokens
    )
    
    # 计算并打印对比结果
    print("\n=== Performance Comparison ===")
    print("Metric\t\tOriginal\tParScale\tDifference")
    print("-" * 60)
    
    mean_orig_loss = np.mean(orig_losses)
    mean_parscale_loss = np.mean(parscale_losses)
    mean_orig_acc = np.mean(orig_accuracies) * 100
    mean_parscale_acc = np.mean(parscale_accuracies) * 100
    
    print(f"Mean Loss\t{mean_orig_loss:.3f}\t\t{mean_parscale_loss:.3f}\t\t{mean_parscale_loss - mean_orig_loss:+.3f}")
    print(f"Mean Accuracy\t{mean_orig_acc:.3f}%\t\t{mean_parscale_acc:.3f}%\t\t{mean_parscale_acc - mean_orig_acc:+.3f}%")
    
    # 比较每个序列的性能
    print("\nSequence-wise Comparison:")
    print("Seq\tOriginal Acc\tParScale Acc\tDifference")
    print("-" * 50)
    for i, (orig_acc, parscale_acc) in enumerate(zip(orig_accuracies, parscale_accuracies)):
        print(f"{i+1}\t{orig_acc:.2%}\t\t{parscale_acc:.2%}\t\t{parscale_acc - orig_acc:+.2%}")

def main():
    """主函数：测试Evo2模型的序列预测性能
    
    支持两种模式：
    1. 仅测试ParScale扩展
    2. 对比原始模型和ParScale扩展的性能
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Forward Pass with ParScale")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'], 
                       default='evo2_7b',
                       help="要测试的模型名称")
    parser.add_argument("--n_parscale", type=int, default=4,
                       help="并行尺度的数量")
    parser.add_argument("--n_tokens", type=int, default=48,
                       help="每个尺度的token数量")
    parser.add_argument("--compare", action="store_true",
                       help="是否进行原始模型和ParScale扩展的对比测试")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # 初始化模型
    model_path = f"/root/EVO/ckpt/{args.model_name}.pt"
    config_path = f"/root/EVO/evo2/configs/{args.model_name.replace('_', '-')}.yml"
    model = Evo2(args.model_name, local_path=model_path)
    
    # 读取测试序列
    sequences = read_prompts('vortex/test/data/output.csv')
    
    if args.compare:
        # 进行原始模型和ParScale扩展的对比测试
        compare_models(
            model=model,
            sequences=sequences,
            n_parscale=args.n_parscale,
            n_tokens=args.n_tokens
        )
    else:
        # 仅测试ParScale扩展
        accuracies, losses = test_forward_pass_parscale(
            model=model,
            sequences=sequences,
            n_parscale=args.n_parscale,
            n_tokens=args.n_tokens
        )
        
        # 计算并打印结果
        mean_loss = np.mean(losses)
        mean_accuracy = np.mean(accuracies) * 100
        print(f"\nMean Loss: {mean_loss:.3f}")
        print(f"Mean Accuracy: {mean_accuracy:.3f}%")
        print(f"ParScale Configuration: n_parscale={args.n_parscale}, n_tokens={args.n_tokens}")

if __name__ == "__main__":
    main() 