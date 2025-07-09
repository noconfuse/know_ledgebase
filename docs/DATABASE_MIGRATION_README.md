# PostgreSQL 数据库迁移工具包

本工具包提供了完整的PostgreSQL数据库导出和导入解决方案，适用于数据库迁移、备份和恢复场景。

## 📁 文件清单

| 文件名 | 描述 | 类型 |
|--------|------|------|
| `export_database.py` | 数据库导出脚本 | Python脚本 |
| `import_database.py` | 数据库导入脚本 | Python脚本 |
| `quick_migration_example.sh` | 快速迁移示例脚本 | Bash脚本 |
| `database_migration_guide.md` | 详细使用指南 | 文档 |
| `DATABASE_MIGRATION_README.md` | 本文件 | 文档 |

## 🚀 快速开始

### 1. 导出数据库
```bash
# 导出完整数据库（推荐）
python export_database.py --compress

# 导出指定表
python export_database.py --tables "table1,table2" --compress

# 仅导出结构
python export_database.py --schema-only
```

### 2. 导入数据库
```bash
# 导入到新数据库
python import_database.py --sql-file backup.sql.gz --create-db

# 替换现有数据库
python import_database.py --sql-file backup.sql.gz --drop-existing --create-db

# 验证模式（不实际导入）
python import_database.py --sql-file backup.sql.gz --dry-run
```

### 3. 运行示例
```bash
# 查看示例脚本帮助
./quick_migration_example.sh --help

# 运行完整示例
./quick_migration_example.sh
```

## 📋 前置要求

- Python 3.6+
- PostgreSQL 客户端工具 (pg_dump, psql, createdb, dropdb)
- 适当的数据库访问权限

### 安装PostgreSQL客户端工具

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-client
```

**CentOS/RHEL:**
```bash
sudo yum install postgresql
```

**macOS:**
```bash
brew install postgresql
```

## 🔧 配置

脚本会自动从 `config.py` 文件读取默认数据库配置，包括：
- 主机地址 (POSTGRES_HOST)
- 端口 (POSTGRES_PORT)
- 数据库名 (POSTGRES_DATABASE)
- 用户名 (POSTGRES_USER)
- 密码 (POSTGRES_PASSWORD)

也可以通过命令行参数覆盖这些默认值。

## 💡 使用场景

### 场景1: 服务器迁移
```bash
# 在源服务器导出
python export_database.py --host source-server --compress

# 传输文件到目标服务器
scp database_export/*.sql.gz user@target-server:/tmp/

# 在目标服务器导入
python import_database.py --host target-server --sql-file /tmp/backup.sql.gz --create-db
```

### 场景2: 定期备份
```bash
# 创建定期备份
python export_database.py --compress --output-dir /backup/$(date +%Y%m%d)
```

### 场景3: 开发环境同步
```bash
# 从生产环境导出特定表
python export_database.py --tables "users,documents" --data-only

# 导入到开发环境
python import_database.py --database dev_db --sql-file backup.sql --create-db
```

## 🛡️ 安全注意事项

1. **密码安全**: 避免在命令行中直接输入密码，使用环境变量或配置文件
2. **文件权限**: 确保备份文件具有适当的权限设置
3. **网络传输**: 使用加密连接传输备份文件
4. **数据验证**: 导入后验证数据完整性

## 📊 性能优化

- 使用 `--compress` 选项减少文件大小
- 对于大型数据库，考虑分表导出
- 在网络传输时使用压缩文件
- 根据需要调整PostgreSQL的并行处理参数

## 🔍 故障排除

### 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `permission denied` | 权限不足 | 检查用户权限，确保有CREATEDB权限 |
| `could not connect` | 连接失败 | 检查主机、端口、防火墙设置 |
| `command not found` | 工具未安装 | 安装PostgreSQL客户端工具 |
| `out of memory` | 内存不足 | 使用压缩选项或分批处理 |

## 📚 详细文档

查看 `database_migration_guide.md` 获取：
- 详细的参数说明
- 完整的使用示例
- 高级配置选项
- 自动化脚本模板
- 性能调优建议

## 🧪 测试

运行示例脚本来测试工具功能：
```bash
# 查看配置
./quick_migration_example.sh --config

# 运行所有示例
./quick_migration_example.sh
```

## 📝 版本信息

- 创建日期: 2024年12月
- 兼容性: PostgreSQL 9.6+
- Python版本: 3.6+
- 测试环境: Ubuntu 20.04+

## 🤝 支持

如果遇到问题：
1. 查看详细文档 `database_migration_guide.md`
2. 运行 `--help` 查看参数说明
3. 使用 `--dry-run` 模式验证操作
4. 检查PostgreSQL日志文件

---

**注意**: 在生产环境使用前，请务必在测试环境中验证所有操作。