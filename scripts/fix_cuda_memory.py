#!/usr/bin/env python3
"""
CUDA内存不足问题解决脚本

此脚本提供了解决CUDA out of memory错误的多种方法：
1. 清理GPU内存
2. 调整嵌入模型批处理大小
3. 检查GPU内存使用情况
4. 提供优化建议
"""

import os
import sys
import torch
import gc
import subprocess
from pathlib import Path

def check_gpu_memory():
    """检查GPU内存使用情况"""
    print("🔍 检查GPU内存使用情况...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查GPU驱动和PyTorch安装")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"📊 检测到 {device_count} 个GPU设备")
    
    for i in range(device_count):
        device = f"cuda:{i}"
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        free = total_memory - cached
        
        print(f"\n🎯 GPU {i} ({props.name}):")
        print(f"   总内存: {total_memory:.2f} GB")
        print(f"   已分配: {allocated:.2f} GB")
        print(f"   已缓存: {cached:.2f} GB")
        print(f"   可用内存: {free:.2f} GB")
        
        if free < 1.0:  # 少于1GB可用内存
            print(f"   ⚠️  警告: GPU {i} 可用内存不足 ({free:.2f} GB)")
    
    return True

def clear_gpu_memory():
    """清理GPU内存"""
    print("\n🧹 清理GPU内存...")
    
    if torch.cuda.is_available():
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        print("✅ GPU内存清理完成")
        
        # 再次检查内存
        check_gpu_memory()
    else:
        print("❌ CUDA不可用，无法清理GPU内存")

def update_embedding_batch_size(batch_size=1):
    """更新嵌入模型批处理大小"""
    print(f"\n⚙️  设置嵌入模型批处理大小为 {batch_size}...")
    
    # 检查.env文件
    env_file = Path(".env")
    env_content = ""
    
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
    
    # 更新或添加EMBEDDING_BATCH_SIZE
    lines = env_content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith('EMBEDDING_BATCH_SIZE='):
            lines[i] = f'EMBEDDING_BATCH_SIZE={batch_size}'
            updated = True
            break
    
    if not updated:
        lines.append(f'EMBEDDING_BATCH_SIZE={batch_size}')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 已设置 EMBEDDING_BATCH_SIZE={batch_size}")
    print("📝 请重启应用以使配置生效")

def kill_gpu_processes():
    """终止占用GPU的进程（谨慎使用）"""
    print("\n⚠️  尝试终止占用GPU的进程...")
    
    try:
        # 使用nvidia-smi查找GPU进程
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            pids = [pid.strip() for pid in result.stdout.split('\n') if pid.strip()]
            
            if pids:
                print(f"🔍 发现 {len(pids)} 个GPU进程: {', '.join(pids)}")
                
                response = input("是否要终止这些进程？这可能会影响其他正在运行的程序 (y/N): ")
                if response.lower() == 'y':
                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-9', pid], check=True)
                            print(f"✅ 已终止进程 {pid}")
                        except subprocess.CalledProcessError:
                            print(f"❌ 无法终止进程 {pid}")
                else:
                    print("❌ 用户取消操作")
            else:
                print("✅ 没有发现GPU进程")
        else:
            print("❌ 无法查询GPU进程，请确保安装了nvidia-smi")
    
    except FileNotFoundError:
        print("❌ nvidia-smi未找到，请确保安装了NVIDIA驱动")

def show_optimization_tips():
    """显示优化建议"""
    print("\n💡 CUDA内存优化建议:")
    print("\n1. 🔧 调整批处理大小:")
    print("   - 当前默认批处理大小为2，可以进一步降低到1")
    print("   - 使用环境变量: EMBEDDING_BATCH_SIZE=1")
    
    print("\n2. 🎯 优化向量存储构建:")
    print("   - 分批处理文档，避免一次性处理大量文件")
    print("   - 考虑使用CPU进行嵌入计算（速度较慢但内存占用少）")
    
    print("\n3. 🧹 定期清理内存:")
    print("   - 在处理大文件前运行此脚本清理GPU内存")
    print("   - 重启应用以完全释放内存")
    
    print("\n4. ⚙️  系统级优化:")
    print("   - 关闭其他占用GPU的程序")
    print("   - 考虑升级GPU内存")
    print("   - 使用更小的嵌入模型")
    
    print("\n5. 🔄 应急方案:")
    print("   - 设置 EMBEDDING_PROVIDER_NAME=siliconflow 使用在线API")
    print("   - 临时禁用GPU: USE_GPU=false")

def main():
    print("🚀 CUDA内存不足问题解决工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 检查GPU内存使用情况")
        print("2. 清理GPU内存")
        print("3. 设置嵌入模型批处理大小")
        print("4. 终止GPU进程（谨慎使用）")
        print("5. 显示优化建议")
        print("6. 退出")
        
        choice = input("\n请输入选项 (1-6): ").strip()
        
        if choice == '1':
            check_gpu_memory()
        elif choice == '2':
            clear_gpu_memory()
        elif choice == '3':
            try:
                batch_size = int(input("请输入批处理大小 (推荐1-2): "))
                if batch_size < 1:
                    print("❌ 批处理大小必须大于0")
                    continue
                update_embedding_batch_size(batch_size)
            except ValueError:
                print("❌ 请输入有效的数字")
        elif choice == '4':
            kill_gpu_processes()
        elif choice == '5':
            show_optimization_tips()
        elif choice == '6':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

if __name__ == "__main__":
    main()