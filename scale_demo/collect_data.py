import argparse
import torch
from pathlib import Path
from evo2 import Evo2
from parscale_wrapper import SimpleParScaleWrapper
from data_collector import DataCollector

def main():
    parser = argparse.ArgumentParser(description="Collect Training Data for ParScale")
    parser.add_argument("--model_name", type=str, default="evo2_1b_base",
                       help="基础模型名称")
    parser.add_argument("--n_parscale", type=int, default=8,
                       help="并行尺度的数量")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="数据保存目录")
    parser.add_argument("--num_sequences", type=int, default=1000,
                       help="收集的序列数量")
    parser.add_argument("--input_file", type=str, 
                       default="/root/EVO/evo2/vortex/test/data/output.csv",
                       help="输入序列文件路径")
    
    args = parser.parse_args()
    
    # 初始化模型
    model_path = f"/root/EVO/ckpt/{args.model_name}.pt"
    config_path = f"/root/EVO/evo2/configs/{args.model_name.replace('_', '-')}.yml"
    model = Evo2(args.model_name, local_path=model_path)
    
    # 创建ParScale包装器
    parscale_model = SimpleParScaleWrapper(model, args.n_parscale)
    
    # 创建数据收集器
    collector = DataCollector(args.data_dir)
    
    # 读取训练序列
    try:
        with open(args.input_file, 'r', encoding='utf-8-sig') as f:
            # 跳过标题行，读取所有序列
            all_sequences = [line.strip().split(',')[0] for line in f.readlines()[1:]]
            # 随机选择指定数量的序列
            import random
            random.shuffle(all_sequences)
            sequences = all_sequences[:args.num_sequences]
            print(f"Loaded {len(sequences)} sequences from {args.input_file}")
    except Exception as e:
        print(f"Error: Failed to read from input file: {str(e)}")
        exit(1)
    
    # 收集数据
    print("Collecting training data...")
    for i, seq in enumerate(sequences):
        # 只保留ATCG序列
        dna_seq = ''.join(c for c in seq if c in 'ATCG')
        
        # 将序列转换为模型输入格式
        input_ids = torch.tensor(model.tokenizer.tokenize(dna_seq), dtype=int).to('cuda:0')
        
        with torch.inference_mode():
            # 使用ParScale进行前向传播
            output = parscale_model.forward(input_ids.unsqueeze(0))
            all_logits, combined_accuracy, all_predictions = output
            
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
                target=input_ids[1:],
                logits=all_logits
            )
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sequences)} sequences")
    
    # 保存数据
    collector.save("training_data.json")
    print(f"Data saved to {args.data_dir}/training_data.json")
    print(f"Total samples collected: {len(collector.data)}")

if __name__ == "__main__":
    main() 