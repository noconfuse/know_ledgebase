#!/bin/bash

# LlamaIndex RAG知识库系统 - 测试运行脚本
# 用于一键运行所有测试用例

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/home/ubuntu/workspace/know_ledgebase"
cd "$PROJECT_ROOT"

# 日志文件
LOG_FILE="test_execution.log"

# 函数：打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_message $RED "错误: $1 命令未找到，请先安装"
        exit 1
    fi
}

# 函数：检查Python包
check_python_package() {
    if ! python -c "import $1" &> /dev/null; then
        print_message $RED "错误: Python包 $1 未安装"
        return 1
    fi
    return 0
}

# 函数：等待服务启动
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_message $YELLOW "等待 $service_name 启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/health" > /dev/null 2>&1; then
            print_message $GREEN "$service_name 已启动"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_message $RED "$service_name 启动超时"
    return 1
}

# 函数：停止服务
stop_services() {
    print_message $YELLOW "停止服务..."
    
    # 查找并杀死相关进程
    pkill -f "document_service.py" 2>/dev/null || true
    pkill -f "rag_service_app.py" 2>/dev/null || true
    pkill -f "start_services.py" 2>/dev/null || true
    
    sleep 2
    print_message $GREEN "服务已停止"
}

# 函数：清理测试数据
cleanup_test_data() {
    print_message $YELLOW "清理测试数据..."
    
    # 清理测试目录
    rm -rf "$PROJECT_ROOT/test_data" 2>/dev/null || true
    rm -rf "$PROJECT_ROOT/quick_test_data" 2>/dev/null || true
    rm -rf "$PROJECT_ROOT/test_output" 2>/dev/null || true
    
    # 清理日志文件
    rm -f "$PROJECT_ROOT/test_workflow.log" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/test_results.json" 2>/dev/null || true
    
    print_message $GREEN "测试数据已清理"
}

# 函数：运行快速测试
run_quick_test() {
    print_message $BLUE "\n=== 运行快速测试 ==="
    
    if python quick_test.py; then
        print_message $GREEN "✅ 快速测试通过"
        return 0
    else
        print_message $RED "❌ 快速测试失败"
        return 1
    fi
}

# 函数：运行API测试
run_api_tests() {
    print_message $BLUE "\n=== 运行API测试 ==="
    
    if python tests/test_api_cases.py; then
        print_message $GREEN "✅ API测试通过"
        return 0
    else
        print_message $RED "❌ API测试失败"
        return 1
    fi
}

# 函数：运行完整工作流程测试
run_workflow_test() {
    print_message $BLUE "\n=== 运行完整工作流程测试 ==="
    
    if python test_workflow.py; then
        print_message $GREEN "✅ 工作流程测试通过"
        return 0
    else
        print_message $RED "❌ 工作流程测试失败"
        return 1
    fi
}

# 函数：显示帮助信息
show_help() {
    echo "LlamaIndex RAG知识库系统 - 测试运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -q, --quick         只运行快速测试"
    echo "  -a, --api           只运行API测试"
    echo "  -w, --workflow      只运行工作流程测试"
    echo "  -c, --cleanup       清理测试数据并退出"
    echo "  -s, --stop          停止服务并退出"
    echo "  --no-cleanup        测试后不清理数据"
    echo "  --no-services       不启动服务（假设服务已运行）"
    echo ""
    echo "默认行为: 运行所有测试"
    echo ""
    echo "示例:"
    echo "  $0                  # 运行所有测试"
    echo "  $0 -q               # 只运行快速测试"
    echo "  $0 --no-services    # 使用已运行的服务进行测试"
}

# 主函数
main() {
    local run_quick=false
    local run_api=false
    local run_workflow=false
    local cleanup_after=true
    local start_services=true
    local cleanup_only=false
    local stop_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -q|--quick)
                run_quick=true
                shift
                ;;
            -a|--api)
                run_api=true
                shift
                ;;
            -w|--workflow)
                run_workflow=true
                shift
                ;;
            -c|--cleanup)
                cleanup_only=true
                shift
                ;;
            -s|--stop)
                stop_only=true
                shift
                ;;
            --no-cleanup)
                cleanup_after=false
                shift
                ;;
            --no-services)
                start_services=false
                shift
                ;;
            *)
                print_message $RED "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果只是清理或停止，直接执行
    if [ "$cleanup_only" = true ]; then
        cleanup_test_data
        exit 0
    fi
    
    if [ "$stop_only" = true ]; then
        stop_services
        exit 0
    fi
    
    # 如果没有指定具体测试，运行所有测试
    if [ "$run_quick" = false ] && [ "$run_api" = false ] && [ "$run_workflow" = false ]; then
        run_quick=true
        run_api=true
        run_workflow=true
    fi
    
    print_message $BLUE "LlamaIndex RAG知识库系统 - 测试执行开始"
    print_message $BLUE "时间: $(date)"
    print_message $BLUE "项目目录: $PROJECT_ROOT"
    
    # 检查环境
    print_message $YELLOW "\n检查环境..."
    check_command python
    check_command curl
    
    # 检查Python包
    local missing_packages=()
    for package in fastapi uvicorn requests pathlib; do
        if ! check_python_package $package; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_message $RED "缺少Python包: ${missing_packages[*]}"
        print_message $YELLOW "请运行: pip install -r requirements.txt"
        exit 1
    fi
    
    print_message $GREEN "环境检查通过"
    
    # 清理之前的测试数据
    cleanup_test_data
    
    # 启动服务
    if [ "$start_services" = true ]; then
        print_message $YELLOW "\n启动服务..."
        
        # 停止可能存在的服务
        stop_services
        
        # 启动新服务
        python start_services.py > "$LOG_FILE" 2>&1 &
        SERVICE_PID=$!
        
        # 等待服务启动
        if ! wait_for_service "http://localhost:8000" "文档服务"; then
            print_message $RED "文档服务启动失败"
            kill $SERVICE_PID 2>/dev/null || true
            exit 1
        fi
        
        if ! wait_for_service "http://localhost:8001" "RAG服务"; then
            print_message $RED "RAG服务启动失败"
            kill $SERVICE_PID 2>/dev/null || true
            exit 1
        fi
        
        print_message $GREEN "所有服务已启动"
    else
        print_message $YELLOW "\n跳过服务启动，使用现有服务"
        
        # 检查服务是否可用
        if ! curl -s "http://localhost:8001/health" > /dev/null; then
            print_message $RED "文档服务不可用"
            exit 1
        fi
        
        if ! curl -s "http://localhost:8002/health" > /dev/null; then
            print_message $RED "RAG服务不可用"
            exit 1
        fi
        
        print_message $GREEN "服务状态检查通过"
    fi
    
    # 运行测试
    local test_results=()
    local overall_success=true
    
    if [ "$run_quick" = true ]; then
        if run_quick_test; then
            test_results+=("快速测试: ✅")
        else
            test_results+=("快速测试: ❌")
            overall_success=false
        fi
    fi
    
    if [ "$run_api" = true ]; then
        if run_api_tests; then
            test_results+=("API测试: ✅")
        else
            test_results+=("API测试: ❌")
            overall_success=false
        fi
    fi
    
    if [ "$run_workflow" = true ]; then
        if run_workflow_test; then
            test_results+=("工作流程测试: ✅")
        else
            test_results+=("工作流程测试: ❌")
            overall_success=false
        fi
    fi
    
    # 停止服务
    if [ "$start_services" = true ]; then
        stop_services
    fi
    
    # 显示测试结果
    print_message $BLUE "\n=== 测试结果汇总 ==="
    for result in "${test_results[@]}"; do
        echo "  $result"
    done
    
    if [ "$overall_success" = true ]; then
        print_message $GREEN "\n🎉 所有测试通过！"
    else
        print_message $RED "\n❌ 部分测试失败，请检查日志"
    fi
    
    print_message $BLUE "\n测试完成时间: $(date)"
    
    # 清理测试数据
    if [ "$cleanup_after" = true ]; then
        cleanup_test_data
    else
        print_message $YELLOW "保留测试数据，位于: $PROJECT_ROOT/test_*"
    fi
    
    # 显示日志文件位置
    if [ -f "$LOG_FILE" ]; then
        print_message $BLUE "服务日志: $PROJECT_ROOT/$LOG_FILE"
    fi
    
    # 退出码
    if [ "$overall_success" = true ]; then
        exit 0
    else
        exit 1
    fi
}

# 捕获中断信号
trap 'print_message $YELLOW "\n测试被中断，正在清理..."; stop_services; cleanup_test_data; exit 130' INT TERM

# 运行主函数
main "$@"