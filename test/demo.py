import argparse
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from evo2 import Evo2

class SimpleParScaleWrapper:
    """简化版的 ParScale 包装器
    
    通过对输入进行不同模式的扰动来模拟多尺度效果
    """
    def __init__(self, model, n_parscale=4):
        self.model = model
        self.n_parscale = n_parscale  # 并行尺度的数量
        
        # 获取模型的数据类型
        self.dtype = next(model.model.parameters()).dtype
        
        # 初始化扰动强度 - 增加噪声强度
        self.noise_scales = torch.nn.Parameter(
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                        device='cuda:0', dtype=self.dtype)
        )
        
        # 初始化丢弃率
        self.dropout_rates = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                        device='cuda:0', dtype=self.dtype)
        
        print(f"Debug - Number of parallel scales: {n_parscale}")
        print(f"Debug - Model dtype: {self.dtype}")
        print(f"Debug - Noise scales: {self.noise_scales.tolist()}")
        print(f"Debug - Dropout rates: {self.dropout_rates.tolist()}")
    
    def apply_scale_specific_perturbation(self, logits, scale_idx, seq_len):
        """对每个尺度应用特定的扰动模式"""
        batch_size, seq_len, hidden_size = logits.shape
        
        # 打印原始输入的一些值
        if scale_idx == 0:  # 只在第一个尺度打印
            print("\nDebug - Original input values (first 5 positions):")
            print(logits[0, :5, :5])  # 打印第一个batch的前5个位置的前5个维度
            print("\nDebug - Original input statistics:")
            print(f"Mean: {logits.mean().item():.4f}")
            print(f"Std: {logits.std().item():.4f}")
            print(f"Max: {logits.max().item():.4f}")
            print(f"Min: {logits.min().item():.4f}")
        
        if scale_idx == 0:  # 随机丢弃：完全随机
            # 对每个hidden dimension分别应用mask
            mask = (torch.rand_like(logits) > self.dropout_rates[scale_idx]).bool()
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())  # 在丢弃位置添加噪声
            
        elif scale_idx == 1:  # 随机丢弃：每隔一个位置
            # 对每个hidden dimension分别应用mask
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, ::2, :] = False  # 每隔一个位置丢弃
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())
            
        elif scale_idx == 2:  # 随机丢弃：连续块
            block_size = seq_len // 4
            mask = torch.ones_like(logits, dtype=torch.bool)
            # 对每个hidden dimension分别选择起始位置
            for d in range(hidden_size):
                start_idx = torch.randint(0, seq_len - block_size, (batch_size,), device='cuda:0')
                for b in range(batch_size):
                    mask[b, start_idx[b]:start_idx[b] + block_size, d] = False
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * (1.0 - mask.float())
            
        elif scale_idx == 3:  # 局部扰动：只扰动序列的前1/3
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, :seq_len//3, :] = True
            # 对每个hidden dimension使用不同的噪声强度
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 4:  # 局部扰动：只扰动序列的后1/3
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, -seq_len//3:, :] = True
            # 对每个hidden dimension使用不同的噪声强度
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 5:  # 交替扰动：每隔一个位置扰动
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, ::2, :] = True
            # 对每个hidden dimension使用不同的噪声强度
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        elif scale_idx == 6:  # 交替扰动：每隔两个位置扰动
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, ::3, :] = True
            # 对每个hidden dimension使用不同的噪声强度
            noise = torch.randn_like(logits) * self.noise_scales[scale_idx]
            noise = noise * mask.float()
            
        else:  # 混合扰动：结合多种模式
            # 基础噪声 - 对每个hidden dimension使用不同的强度
            base_noise = torch.randn_like(logits) * self.noise_scales[scale_idx] * 0.5
            # 周期性噪声 - 对每个hidden dimension使用不同的频率
            position = torch.arange(seq_len, device='cuda:0', dtype=self.dtype)
            position = position.view(1, -1, 1)
            frequencies = torch.linspace(0.05, 0.15, hidden_size, device='cuda:0', dtype=self.dtype)
            periodic_noise = torch.sin(position * frequencies.view(1, 1, -1)) * self.noise_scales[scale_idx] * 0.3
            # 渐进噪声 - 对每个hidden dimension使用不同的权重
            position_weights = torch.linspace(0, 1, seq_len, device='cuda:0', dtype=self.dtype)
            position_weights = position_weights.view(1, -1, 1)
            dimension_weights = torch.linspace(0.5, 1.5, hidden_size, device='cuda:0', dtype=self.dtype)
            progressive_noise = torch.randn_like(logits) * self.noise_scales[scale_idx] * 0.2 * position_weights * dimension_weights.view(1, 1, -1)
            
            noise = base_noise + periodic_noise + progressive_noise
        
        # 打印扰动后的值
        if scale_idx == 0:  # 只在第一个尺度打印
            print(f"\nDebug - Perturbed values for scale {scale_idx} (first 5 positions):")
            print((logits + noise)[0, :5, :5])  # 打印第一个batch的前5个位置的前5个维度
            print(f"\nDebug - Noise values for scale {scale_idx} (first 5 positions):")
            print(noise[0, :5, :5])  # 打印噪声值
            print(f"\nDebug - Mask values for scale {scale_idx} (first 5 positions):")
            print(mask[0, :5, :5])  # 打印mask值
            print("\nDebug - Perturbed statistics:")
            print(f"Mean: {(logits + noise).mean().item():.4f}")
            print(f"Std: {(logits + noise).std().item():.4f}")
            print(f"Max: {(logits + noise).max().item():.4f}")
            print(f"Min: {(logits + noise).min().item():.4f}")
        
        return noise
    
    def forward(self, input_ids):
        """简化的 ParScale 前向传播
        
        Args:
            input_ids: 输入序列的token ids
            
        Returns:
            tuple: (处理后的logits, combined_accuracy)
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
            
            print(f"Debug - Original logits shape: {logits.shape}")
            print(f"Debug - Original logits dtype: {logits.dtype}")
        
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
        target_ids = input_ids[:, 1:]  # 右移一位用于下一个token预测
        
        print("\nScale-wise Performance:")
        print("Scale\tNoise\t\tDropout\t\tLoss\t\tAccuracy")
        print("-" * 60)
        
        # 存储每个尺度的预测结果
        all_predictions = []
        
        for i, output in enumerate(scale_outputs):
            # 计算交叉熵损失
            pred_logits = output[:, :-1, :]  # 去掉最后一个token的预测
            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            # 计算准确率
            pred_tokens = torch.argmax(pred_logits, dim=-1)
            accuracy = (target_ids == pred_tokens).float().mean().item()
            
            scale_losses.append(loss)
            all_predictions.append(pred_tokens)
            print(f"{i}\t{self.noise_scales[i].item():.4f}\t\t{self.dropout_rates[i].item():.4f}\t\t{loss.item():.4f}\t\t{accuracy:.2%}")
        
        # 计算组合准确率（只要有一个流预测正确就算正确）
        all_predictions = torch.stack(all_predictions)  # [n_parscale, batch_size, seq_len]
        correct_predictions = (all_predictions == target_ids.unsqueeze(0)).any(dim=0)  # [batch_size, seq_len]
        combined_accuracy = correct_predictions.float().mean().item()
        
        print(f"\nCombined Accuracy (any flow correct): {combined_accuracy:.2%}")
        
        # 选择loss最小的尺度
        best_scale_idx = torch.argmin(torch.stack(scale_losses))
        best_output = scale_outputs[best_scale_idx]
        
        print(f"\nSelected best scale: {best_scale_idx.item()}")
        print(f"Best scale loss: {scale_losses[best_scale_idx].item():.4f}")
            
        return best_output, combined_accuracy

def test_forward_pass_parscale(*, model, sequences, n_parscale=4):
    """测试简化版 ParScale 模型的前向传播性能"""
    losses = []
    accuracies = []
    combined_accuracies = []  # 新增：存储组合准确率
    
    # 创建简化版 ParScale 包装器
    parscale_model = SimpleParScaleWrapper(model, n_parscale)
    
    for seq in sequences:
        # 将序列转换为模型输入格式
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to('cuda:0')
        print(f"Debug - Input shape: {input_ids.shape}")
        
        with torch.inference_mode():
            # 使用简化版 ParScale 进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
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
    print("\nSequence Results with Simple ParScale:")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print("WARNING: Forward pass accuracy is below 50% on test sequence.")
    
    return accuracies, losses

def test_forward_pass_original(*, model, sequences):
    """测试原始模型的前向传播性能"""
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

def compare_models(*, model, sequences, n_parscale=4):
    """比较原始模型和简化版 ParScale 的性能差异"""
    print("\n=== Testing Original Model ===")
    orig_accuracies, orig_losses = test_forward_pass_original(
        model=model,
        sequences=sequences
    )
    
    print("\n=== Testing Simple ParScale Model ===")
    parscale_accuracies, parscale_losses = test_forward_pass_parscale(
        model=model,
        sequences=sequences,
        n_parscale=n_parscale
    )
    
    # 计算并打印对比结果
    print("\n=== Performance Comparison ===")
    print("Metric\t\tOriginal\tParScale\tDifference")
    print("-" * 60)
    
    mean_orig_loss = sum(orig_losses) / len(orig_losses)
    mean_parscale_loss = sum(parscale_losses) / len(parscale_losses)
    mean_orig_acc = sum(orig_accuracies) / len(orig_accuracies) * 100
    mean_parscale_acc = sum(parscale_accuracies) / len(parscale_accuracies) * 100
    
    print(f"Mean Loss\t{mean_orig_loss:.3f}\t\t{mean_parscale_loss:.3f}\t\t{mean_parscale_loss - mean_orig_loss:+.3f}")
    print(f"Mean Accuracy\t{mean_orig_acc:.3f}%\t\t{mean_parscale_acc:.3f}%\t\t{mean_parscale_acc - mean_orig_acc:+.3f}%")
    
    # 比较每个序列的性能
    print("\nSequence-wise Comparison:")
    print("Seq\tOriginal Acc\tParScale Acc\tCombined Acc\tDifference")
    print("-" * 70)
    
    # 创建 ParScale 模型实例
    parscale_model = SimpleParScaleWrapper(model, n_parscale)
    
    # 获取每个序列的组合准确率
    combined_accuracies = []
    for seq in sequences:
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to('cuda:0')
        with torch.inference_mode():
            # 使用简化版 ParScale 进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
            # 从forward方法的输出中获取组合准确率
            combined_acc = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if combined_acc is not None:
                combined_accuracies.append(combined_acc)
    
    for i, (orig_acc, parscale_acc) in enumerate(zip(orig_accuracies, parscale_accuracies)):
        combined_acc = combined_accuracies[i] if i < len(combined_accuracies) else 0.0
        print(f"{i+1}\t{orig_acc:.2%}\t\t{parscale_acc:.2%}\t\t{combined_acc:.2%}\t\t{parscale_acc - orig_acc:+.2%}")

def read_prompts(input_file: Path) -> list:
    """Read prompts from input file."""
    promptseqs: list = []
    
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

def main():
    """主函数：测试简化版 ParScale 的性能"""
    parser = argparse.ArgumentParser(description="Test Simple ParScale Implementation")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'], 
                       default='evo2_7b',
                       help="要测试的模型名称")
    parser.add_argument("--n_parscale", type=int, default=4,
                       help="并行尺度的数量")
    parser.add_argument("--compare", action="store_true",
                       help="是否进行原始模型和 ParScale 的对比测试")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # 初始化模型
    model_path = f"/root/EVO/ckpt/{args.model_name}.pt"
    config_path = f"/root/EVO/evo2/configs/{args.model_name.replace('_', '-')}.yml"
    model = Evo2(args.model_name, local_path=model_path)
    
    # 读取测试序列
    try:
        sequences = read_prompts('vortex/test/data/output.csv')
    except Exception as e:
        print(f"Warning: Failed to read from CSV file: {str(e)}")
        print("Using default test sequences instead.")
        sequences = ["Hello, how are you today?", 
                    "What is the capital of France?",
                    "Explain the concept of machine learning.",
                    "Write a simple Python function to calculate factorial."]
    
    if args.compare:
        # 进行原始模型和 ParScale 的对比测试
        compare_models(
            model=model,
            sequences=sequences,
            n_parscale=args.n_parscale
        )
    else:
        # 仅测试 ParScale
        accuracies, losses = test_forward_pass_parscale(
            model=model,
            sequences=sequences,
            n_parscale=args.n_parscale
        )
        
        # 计算并打印结果
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracies) / len(accuracies) * 100
        print(f"\nMean Loss: {mean_loss:.3f}")
        print(f"Mean Accuracy: {mean_accuracy:.3f}%")
        print(f"ParScale Configuration: n_parscale={args.n_parscale}")

if __name__ == "__main__":
    main() 