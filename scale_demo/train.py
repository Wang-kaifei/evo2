import argparse
import torch
from pathlib import Path
from evo2 import Evo2
from parscale_wrapper import SimpleParScaleWrapper
from reward_model import RewardModel, train_reward_model
from data_collector import DataCollector, DataLoader

def collect_training_data(model, sequences, n_parscale=8, save_dir="data"):
    """收集训练数据
    
    Args:
        model: 基础模型
        sequences: 训练序列列表
        n_parscale: 并行尺度的数量
        save_dir: 数据保存目录
    """
    # 创建ParScale包装器
    parscale_model = SimpleParScaleWrapper(model, n_parscale)
    
    # 创建数据收集器
    collector = DataCollector(save_dir)
    
    # 收集数据
    for seq in sequences:
        # 只保留ATCG序列
        dna_seq = ''.join(c for c in seq if c in 'ATCG')
        
        # 将序列转换为模型输入格式
        input_ids = torch.tensor(model.tokenizer.tokenize(dna_seq), dtype=int).to('cuda:0')
        
        # 打印tokenizer的详细信息
        # print("\nSequence:", dna_seq)
        # print("Token IDs:", input_ids.tolist())
        
        with torch.inference_mode():
            # 使用ParScale进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
            logits, combined_accuracy, all_predictions = output
            
            # 准备特征
            scale_features = torch.stack([
                parscale_model.noise_scales,
                parscale_model.dropout_rates
            ], dim=1)  # [n_parscale, 2]
            
            # 收集数据
            collector.collect(
                context=input_ids.unsqueeze(0),
                predictions=all_predictions,
                scale_features=scale_features,
                target=input_ids[1:]
            )
    
    # 保存数据
    collector.save("training_data.json")
    return collector.data

def main():
    parser = argparse.ArgumentParser(description="Train Reward Model for ParScale")
    parser.add_argument("--model_name", type=str, default="evo2_1b_base",
                       help="基础模型名称")
    parser.add_argument("--n_parscale", type=int, default=8,
                       help="并行尺度的数量")
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="Reward Model的隐藏层大小")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="训练批次大小")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="数据保存目录")
    parser.add_argument("--collect_only", action="store_true",
                       help="只收集数据，不训练模型")
    
    args = parser.parse_args()
    
    # 初始化模型
    model_path = f"/root/EVO/ckpt/{args.model_name}.pt"
    config_path = f"/root/EVO/evo2/configs/{args.model_name.replace('_', '-')}.yml"
    model = Evo2(args.model_name, local_path=model_path)
    
    # 读取训练序列
    try:
        with open('/root/EVO/evo2/vortex/test/data/output.csv', 'r', encoding='utf-8-sig') as f:
            # 跳过标题行，只读取第一列（DNA序列）
            sequences = [line.strip().split(',')[0] for line in f.readlines()[1:11]]
    except Exception as e:
        print(f"Warning: Failed to read from CSV file: {str(e)}")
        print("Using default test sequences instead.")
        exit()
    
    # 收集训练数据
    print("Collecting training data...")
    data = collect_training_data(model, sequences, args.n_parscale, args.data_dir)
    
    if args.collect_only:
        return
    
    # 准备数据加载器
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_loader = DataLoader(train_data, args.batch_size)
    val_loader = DataLoader(val_data, args.batch_size)
    
    # 创建Reward Model
    reward_model = RewardModel(
        vocab_size=model.tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        n_parscale=args.n_parscale
    ).to('cuda:0')
    
    # 训练模型
    print("Training reward model...")
    train_reward_model(
        model=reward_model,
        train_data=train_loader,
        val_data=val_loader,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # 保存模型
    save_path = Path(args.data_dir) / "reward_model.pt"
    torch.save(reward_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main() 