#!/bin/bash
# PostgreSQL 数据库迁移快速示例脚本
# 此脚本展示了如何使用 export_database.py 和 import_database.py 进行数据库迁移

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置变量
SOURCE_HOST="localhost"
SOURCE_PORT="5432"
SOURCE_DB="knowledge_base"
SOURCE_USER="postgres"
SOURCE_PASSWORD="postgres"

TARGET_HOST="localhost"
TARGET_PORT="5432"
TARGET_DB="knowledge_base_backup"
TARGET_USER="postgres"
TARGET_PASSWORD="postgres"

BACKUP_DIR="./database_backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 显示配置信息
show_config() {
    print_info "=== 数据库迁移配置 ==="
    echo "源数据库: ${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_PORT}/${SOURCE_DB}"
    echo "目标数据库: ${TARGET_USER}@${TARGET_HOST}:${TARGET_PORT}/${TARGET_DB}"
    echo "备份目录: ${BACKUP_DIR}"
    echo "时间戳: ${TIMESTAMP}"
    echo
}

# 检查脚本是否存在
check_scripts() {
    print_info "检查必要的脚本文件..."
    
    if [ ! -f "export_database.py" ]; then
        print_error "export_database.py 不存在"
        exit 1
    fi
    
    if [ ! -f "import_database.py" ]; then
        print_error "import_database.py 不存在"
        exit 1
    fi
    
    print_success "脚本文件检查完成"
}

# 示例1: 完整数据库导出
example_full_export() {
    print_info "=== 示例1: 完整数据库导出 ==="
    
    print_info "导出完整数据库（包含结构和数据）..."
    python export_database.py \
        --host "$SOURCE_HOST" \
        --port "$SOURCE_PORT" \
        --database "$SOURCE_DB" \
        --user "$SOURCE_USER" \
        --password "$SOURCE_PASSWORD" \
        --output-dir "$BACKUP_DIR" \
        --compress
    
    if [ $? -eq 0 ]; then
        print_success "完整数据库导出成功"
    else
        print_error "完整数据库导出失败"
        return 1
    fi
}

# 示例2: 仅导出数据库结构
example_schema_export() {
    print_info "=== 示例2: 仅导出数据库结构 ==="
    
    print_info "导出数据库结构..."
    python export_database.py \
        --host "$SOURCE_HOST" \
        --port "$SOURCE_PORT" \
        --database "$SOURCE_DB" \
        --user "$SOURCE_USER" \
        --password "$SOURCE_PASSWORD" \
        --output-dir "$BACKUP_DIR" \
        --schema-only
    
    if [ $? -eq 0 ]; then
        print_success "数据库结构导出成功"
    else
        print_error "数据库结构导出失败"
        return 1
    fi
}

# 示例3: 导出指定表
example_table_export() {
    print_info "=== 示例3: 导出指定表 ==="
    
    # 首先获取数据库中的表列表
    print_info "获取数据库表信息..."
    python export_database.py \
        --host "$SOURCE_HOST" \
        --port "$SOURCE_PORT" \
        --database "$SOURCE_DB" \
        --user "$SOURCE_USER" \
        --password "$SOURCE_PASSWORD" \
        --info
    
    # 导出特定表（这里假设存在这些表）
    print_info "导出指定表..."
    python export_database.py \
        --host "$SOURCE_HOST" \
        --port "$SOURCE_PORT" \
        --database "$SOURCE_DB" \
        --user "$SOURCE_USER" \
        --password "$SOURCE_PASSWORD" \
        --output-dir "$BACKUP_DIR" \
        --tables "parse_tasks,vector_store_tasks" \
        --compress
    
    if [ $? -eq 0 ]; then
        print_success "指定表导出成功"
    else
        print_warning "指定表导出失败（可能是表不存在）"
    fi
}

# 示例4: 数据库导入（干运行模式）
example_dry_run_import() {
    print_info "=== 示例4: 数据库导入（干运行模式） ==="
    
    # 查找最新的备份文件
    LATEST_BACKUP=$(find "$BACKUP_DIR" -name "*.sql*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_BACKUP" ]; then
        print_warning "未找到备份文件，跳过导入示例"
        return 0
    fi
    
    print_info "使用备份文件: $LATEST_BACKUP"
    print_info "执行干运行导入（仅验证，不实际导入）..."
    
    python import_database.py \
        --host "$TARGET_HOST" \
        --port "$TARGET_PORT" \
        --database "$TARGET_DB" \
        --user "$TARGET_USER" \
        --password "$TARGET_PASSWORD" \
        --sql-file "$LATEST_BACKUP" \
        --dry-run \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "干运行导入验证成功"
    else
        print_error "干运行导入验证失败"
        return 1
    fi
}

# 示例5: 实际数据库导入（可选）
example_real_import() {
    print_info "=== 示例5: 实际数据库导入 ==="
    
    read -p "是否要执行实际的数据库导入？这将创建新数据库 '$TARGET_DB' (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过实际导入"
        return 0
    fi
    
    # 查找最新的完整备份文件
    LATEST_FULL_BACKUP=$(find "$BACKUP_DIR" -name "*_full_*.sql*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_FULL_BACKUP" ]; then
        print_error "未找到完整备份文件"
        return 1
    fi
    
    print_info "使用完整备份文件: $LATEST_FULL_BACKUP"
    print_info "创建新数据库并导入数据..."
    
    python import_database.py \
        --host "$TARGET_HOST" \
        --port "$TARGET_PORT" \
        --database "$TARGET_DB" \
        --user "$TARGET_USER" \
        --password "$TARGET_PASSWORD" \
        --sql-file "$LATEST_FULL_BACKUP" \
        --create-db \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "数据库导入成功"
        print_info "新数据库 '$TARGET_DB' 已创建并导入数据"
    else
        print_error "数据库导入失败"
        return 1
    fi
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    # 这里可以添加清理逻辑
    print_success "清理完成"
}

# 主函数
main() {
    print_info "PostgreSQL 数据库迁移示例脚本"
    print_info "此脚本演示如何使用 export_database.py 和 import_database.py"
    echo
    
    show_config
    check_scripts
    
    # 创建备份目录
    mkdir -p "$BACKUP_DIR"
    
    # 执行示例
    example_full_export || print_warning "完整导出示例失败"
    echo
    
    example_schema_export || print_warning "结构导出示例失败"
    echo
    
    example_table_export || print_warning "表导出示例失败"
    echo
    
    example_dry_run_import || print_warning "干运行导入示例失败"
    echo
    
    example_real_import || print_warning "实际导入示例失败"
    echo
    
    print_success "所有示例执行完成"
    print_info "备份文件保存在: $BACKUP_DIR"
    
    # 显示备份文件列表
    if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR)" ]; then
        print_info "生成的备份文件:"
        ls -lh "$BACKUP_DIR"
    fi
}

# 信号处理
trap cleanup EXIT

# 检查参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "PostgreSQL 数据库迁移示例脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  --config       仅显示配置信息"
    echo
    echo "此脚本将演示以下操作:"
    echo "  1. 完整数据库导出（压缩）"
    echo "  2. 仅导出数据库结构"
    echo "  3. 导出指定表"
    echo "  4. 干运行导入验证"
    echo "  5. 实际数据库导入（可选）"
    echo
    echo "请在运行前修改脚本中的数据库连接配置"
    exit 0
fi

if [ "$1" = "--config" ]; then
    show_config
    exit 0
fi

# 运行主函数
main