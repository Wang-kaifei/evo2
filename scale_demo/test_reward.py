import torch
import numpy as np
import pandas as pd
from pathlib import Path
from evo2 import Evo2
from parscale_wrapper import SimpleParScaleWrapper
from reward_model import RewardModel
from data_collector import DataCollector
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def evaluate_predictions(predictions, targets, n_parscale):
    """评估预测结果
    
    Args:
        predictions: 预测的token ID [batch_size, seq_len-1]
        targets: 目标token ID [batch_size, seq_len-1]
        n_parscale: 并行scale数量
        
    Returns:
        metrics: 评估指标字典
    """
    # 计算每个位置的准确率
    position_accuracy = (predictions == targets).float().mean(dim=0)
    
    # 计算总体准确率
    overall_accuracy = (predictions == targets).float().mean()
    
    return {
        'position_accuracy': position_accuracy.cpu().numpy(),
        'overall_accuracy': overall_accuracy.item()
    }

def plot_metrics(metrics, save_dir):
    """绘制评估指标图表
    
    Args:
        metrics: 评估指标字典
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制位置准确率
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['position_accuracy'])
    plt.title('Position-wise Accuracy')
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.savefig(save_dir / 'position_accuracy.png')
    plt.close()
    
    # 绘制总体准确率
    plt.figure(figsize=(10, 5))
    plt.bar(['Overall'], [metrics['overall_accuracy']])
    plt.title('Overall Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(save_dir / 'overall_accuracy.png')
    plt.close()

def test_original_model(model, sequences):
    """测试原始模型的前向传播性能
    
    Args:
        model: Evo2模型实例
        sequences: 测试序列列表
        
    Returns:
        accuracies: 每个序列的准确率列表
        losses: 每个序列的损失列表
    """
    losses = []
    accuracies = []
    all_predictions = []
    all_targets = []
    
    print("\n=== Testing Original Model ===")
    for i, seq in enumerate(tqdm(sequences)):
        # 只保留ATCG序列
        dna_seq = ''.join(c for c in seq if c in 'ATCG')
        
        # 将序列转换为ASCII码
        input_ids = torch.tensor([ord(c) for c in dna_seq], dtype=int).to('cuda:0')
        
        with torch.inference_mode():
            # 原始模型的前向传播
            output = model(input_ids.unsqueeze(0))
            if isinstance(output, tuple):
                logits = output[0]
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                logits = output
            
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
            
            all_predictions.append(pred_tokens.cpu().numpy())
            all_targets.append(target_ids.cpu().numpy())
            
            # 打印每个序列的结果
            print(f"\nSequence {i+1}:")
            print(f"Accuracy: {100*accuracy:.2f}%")
    
    # 计算总体结果
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    overall_accuracy = 100 * (all_predictions == all_targets).mean()
    
    print(f"\nOriginal Model Overall Results:")
    print(f"Total sequences: {len(sequences)}")
    print(f"Total tokens: {len(all_targets)}")
    print(f"Correct predictions: {(all_predictions == all_targets).sum()}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    return accuracies, losses

def main():
    # 初始化基础模型
    model_name = "evo2_1b_base"
    model_path = f"/root/EVO/ckpt/{model_name}.pt"
    model = Evo2(model_name, local_path=model_path)
    
    # 加载reward模型
    reward_model = RewardModel(
        vocab_size=128,  # ASCII码范围是0-127
        hidden_size=768,
        n_parscale=8
    ).to('cuda:0')
    
    reward_model.load_state_dict(torch.load("data/reward_model.pt"))
    reward_model.eval()
    
    # 创建ParScale包装器
    parscale_model = SimpleParScaleWrapper(model, n_parscale=8)
    
    # 创建数据收集器
    data_collector = DataCollector(save_dir="data")
    
    # 加载测试数据
    test_file = "/root/EVO/evo2/vortex/test/data/prompts.csv"
    test_data = pd.read_csv(test_file)
    test_sequences = test_data['Sequence'].values[:]  # 取前10个样本
    
    print(f"Testing on {len(test_sequences)} sequences...")
    
    print("\n=== Testing Reward Model with ParScale ===")
    
    # 准备存储结果
    results = []
    all_predictions = []
    all_targets = []
    
    for i, seq in enumerate(tqdm(test_sequences)):
        # 只保留ATCG序列
        dna_seq = ''.join(c for c in seq if c in 'ATCG')
        
        # 将序列转换为ASCII码
        input_ids = torch.tensor([ord(c) for c in dna_seq], dtype=int).to('cuda:0')
        
        with torch.inference_mode():
            # 使用ParScale进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
            logits, combined_accuracy, all_predictions_scale = output
            
            # 获取序列长度和目标序列
            seq_len = input_ids.size(0)
            target = input_ids[1:]  # [seq_len-1]
            
            # 准备特征，确保使用正确的数据类型
            scale_features = torch.stack([
                parscale_model.noise_scales,
                parscale_model.dropout_rates
            ], dim=1).to(dtype=torch.bfloat16)  # [n_parscale, 2]
            
            # 计算prediction_scores
            prediction_scores = torch.zeros(8, 1, seq_len-1, device='cuda:0', dtype=torch.bfloat16)
            for scale in range(8):
                # 计算exact match
                exact_match = (all_predictions_scale[scale, 0] == target).float()
                
                # 计算confidence score
                scale_logits = logits[scale, 0, :-1]  # [seq_len-1, vocab_size]
                target_logits = scale_logits[torch.arange(seq_len-1), target]
                probs = F.softmax(scale_logits, dim=-1)
                target_probs = probs[torch.arange(seq_len-1), target]
                
                # 组合scores
                prediction_scores[scale, 0] = exact_match + target_probs
            
            # 使用reward模型预测
            rewards = reward_model(
                context=input_ids.unsqueeze(0),
                predictions=all_predictions_scale,
                scale_features=scale_features,
                prediction_scores=prediction_scores
            )  # [n_parscale, batch_size, seq_len-1]
            
            # 选择每个位置reward分数最高的预测
            best_predictions = torch.zeros(1, seq_len-1, dtype=torch.long, device='cuda:0')
            for t in range(seq_len-1):
                best_scale = rewards[:, 0, t].argmax().item()
                best_predictions[0, t] = all_predictions_scale[best_scale, 0, t]
            
            # 计算准确率
            correct = (best_predictions[0] == target).sum().item()
            accuracy = 100 * correct / (seq_len - 1)
            
            # 存储结果
            results.append({
                'sequence': dna_seq,
                'accuracy': accuracy,
                'best_scales': rewards.argmax(dim=0)[0].cpu().numpy().tolist()
            })
            
            all_predictions.append(best_predictions[0].cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # 打印每个序列的结果
            print(f"\nSequence {i+1}:")
            print(f"Accuracy: {accuracy:.2f}%")
            
            # 打印每个scale的使用统计
            scale_counts = torch.bincount(rewards.argmax(dim=0)[0], minlength=8)
            print("\nScale usage statistics:")
            for scale in range(8):
                print(f"Scale {scale}: {scale_counts[scale].item()} times ({100*scale_counts[scale].item()/(seq_len-1):.1f}%)")
            
            # 计算每个scale的准确率（当被选择时）
            print("\nScale-wise accuracy (when selected):")
            for scale in range(8):
                # 找出使用该scale的位置
                scale_mask = (rewards.argmax(dim=0)[0] == scale)
                if scale_mask.any():
                    scale_preds = best_predictions[0][scale_mask]
                    scale_targets = target[scale_mask]
                    scale_acc = (scale_preds == scale_targets).float().mean().item()
                    print(f"Scale {scale}: {100*scale_acc:.2f}%")
            
            # 计算如果所有位置都使用同一个scale时的准确率
            print("\nAccuracy when using single scale for all positions:")
            for scale in range(8):
                # 使用该scale的所有预测
                scale_preds = all_predictions_scale[scale, 0]  # [seq_len-1]
                scale_acc = (scale_preds == target).float().mean().item()
                print(f"Scale {scale}: {100*scale_acc:.2f}%")
    
    # 计算总体评估指标
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # 计算token预测准确率
    token_accuracy = 100 * (all_predictions == all_targets).mean()
    
    # 保存详细结果
    results_df = pd.DataFrame(results)
    # 创建保存目录
    save_dir = Path("data/test_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_dir / "detailed_results.csv", index=False)
    
    # 首先测试原始模型
    orig_accuracies, orig_losses = test_original_model(model, test_sequences)
    
    # 打印总体结果
    print(f"\nReward Model Overall Results:")
    print(f"Total sequences: {len(test_sequences)}")
    print(f"Total tokens: {len(all_targets)}")
    print(f"Correct predictions: {(all_predictions == all_targets).sum()}")
    print(f"Token Prediction Accuracy: {token_accuracy:.2f}%")
    
    # 打印模型对比结果
    print("\n=== Model Comparison ===")
    print("Model\t\tAccuracy\tImprovement")
    print("-" * 40)
    print(f"Original\t{100*sum(orig_accuracies)/len(orig_accuracies):.2f}%")
    print(f"Reward Model\t{token_accuracy:.2f}%\t{token_accuracy - 100*sum(orig_accuracies)/len(orig_accuracies):+.2f}%")

if __name__ == "__main__":
    main() 