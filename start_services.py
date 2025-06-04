#!/usr/bin/env python3
"""
启动脚本 - 同时启动文档解析服务和RAG检索服务
"""

import asyncio
import subprocess
import sys
import signal
import time
from pathlib import Path
import logging
from config import settings
from utils.logging_config import setup_logging, get_logger

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        self.processes = []
        self.running = False
    
    def start_service(self, service_name: str, module_path: str, host: str, port: int):
        """启动单个服务"""
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                f"{module_path}:app",
                "--host", host,
                "--port", str(port),
                "--reload",
                "--log-level", "info"
            ]
            
            logger.info(f"Starting {service_name} on {host}:{port}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append({
                'name': service_name,
                'process': process,
                'host': host,
                'port': port
            })
            
            logger.info(f"{service_name} started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
            return None
    
    def start_all_services(self):
        """启动所有服务"""
        try:
            # 启动文档解析服务
            doc_service = self.start_service(
                "Document Service",
                "apps.document_service",
                "0.0.0.0",
                8001
            )
            
            if not doc_service:
                logger.error("Failed to start Document Service")
                return False
            
            # 等待一下再启动下一个服务
            time.sleep(2)
            
            # 启动RAG检索服务
            rag_service = self.start_service(
                "RAG Service",
                "apps.rag_service_app",
                "0.0.0.0",
                8002
            )
            
            if not rag_service:
                logger.error("Failed to start RAG Service")
                self.stop_all_services()
                return False
            
            self.running = True
            logger.info("All services started successfully!")
            
            # 打印服务信息
            self.print_service_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            self.stop_all_services()
            return False
    
    def print_service_info(self):
        """打印服务信息"""
        print("\n" + "="*60)
        print("🚀 服务启动成功!")
        print("="*60)
        
        for service in self.processes:
            print(f"📋 {service['name']}:")
            print(f"   URL: http://{service['host']}:{service['port']}")
            print(f"   PID: {service['process'].pid}")
            print(f"   文档: http://{service['host']}:{service['port']}/docs")
            print()
        
        print("💡 使用说明:")
        print("   - 文档解析服务 (端口 8001): 上传和解析文档")
        print("   - RAG检索服务 (端口 8002): 向量检索和对话")
        print("   - 按 Ctrl+C 停止所有服务")
        print("="*60 + "\n")
    
    def stop_all_services(self):
        """停止所有服务"""
        logger.info("Stopping all services...")
        
        for service in self.processes:
            try:
                process = service['process']
                if process.poll() is None:  # 进程还在运行
                    logger.info(f"Stopping {service['name']} (PID: {process.pid})")
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {service['name']}")
                        process.kill()
                        process.wait()
                    
                    logger.info(f"{service['name']} stopped")
                    
            except Exception as e:
                logger.error(f"Error stopping {service['name']}: {e}")
        
        self.processes.clear()
        self.running = False
        logger.info("All services stopped")
    
    def monitor_services(self):
        """监控服务状态"""
        while self.running:
            try:
                for service in self.processes:
                    process = service['process']
                    if process.poll() is not None:  # 进程已结束
                        logger.error(f"{service['name']} has stopped unexpectedly")
                        self.running = False
                        break
                
                time.sleep(5)  # 每5秒检查一次
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
                break
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"Received signal {signum}")
        self.running = False
        self.stop_all_services()
        sys.exit(0)

def check_dependencies():
    """检查依赖"""
    try:
        import fastapi
        import uvicorn
        import llama_index
        logger.info("Dependencies check passed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("🔧 LlamaIndex RAG 服务启动器")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查当前目录
    if not Path("config.py").exists():
        logger.error("config.py not found. Please run from project root directory.")
        sys.exit(1)
    
    # 创建服务管理器
    manager = ServiceManager()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        # 启动所有服务
        if not manager.start_all_services():
            logger.error("Failed to start services")
            sys.exit(1)
        
        # 监控服务
        manager.monitor_services()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        manager.stop_all_services()

if __name__ == "__main__":
    main()