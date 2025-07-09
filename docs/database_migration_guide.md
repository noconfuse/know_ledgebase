# PostgreSQL 数据库迁移指南

本指南提供了完整的PostgreSQL数据库导出和导入解决方案，包括两个Python脚本：`export_database.py` 和 `import_database.py`。

## 前置要求

### 1. 安装PostgreSQL客户端工具

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql-client
```

**CentOS/RHEL:**
```bash
sudo yum install postgresql
# 或者对于较新版本
sudo dnf install postgresql
```

**macOS:**
```bash
brew install postgresql
```

### 2. Python环境
确保Python 3.6+已安装，脚本使用标准库，无需额外依赖。

## 导出数据库 (export_database.py)

### 基本用法

#### 1. 导出完整数据库
```bash
# 使用默认配置导出
python export_database.py

# 指定数据库连接参数
python export_database.py --host localhost --port 5432 --database knowledge_base --user postgres --password your_password

# 导出并压缩
python export_database.py --compress
```

#### 2. 仅导出数据库结构
```bash
python export_database.py --schema-only
```

#### 3. 仅导出数据
```bash
# 导出所有表的数据
python export_database.py --data-only

# 导出指定表的数据
python export_database.py --data-only --tables "table1,table2,table3"
```

#### 4. 导出指定表（结构+数据）
```bash
python export_database.py --tables "users,documents,vector_store_tasks"
```

#### 5. 查看数据库信息
```bash
python export_database.py --info
```

### 导出选项说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 数据库主机地址 | localhost |
| `--port` | 数据库端口 | 5432 |
| `--database` | 数据库名称 | knowledge_base |
| `--user` | 数据库用户名 | postgres |
| `--password` | 数据库密码 | postgres |
| `--output-dir` | 导出文件保存目录 | ./database_export |
| `--tables` | 指定要导出的表名（逗号分隔） | 无（导出全库） |
| `--data-only` | 仅导出数据，不导出结构 | False |
| `--schema-only` | 仅导出结构，不导出数据 | False |
| `--compress` | 压缩导出文件 | False |
| `--info` | 显示数据库信息 | False |

### 导出文件命名规则

- 完整数据库：`{database}_full_{timestamp}.sql`
- 仅结构：`{database}_schema_{timestamp}.sql`
- 仅数据：`{database}_data_{timestamp}.sql`
- 指定表：`{database}_tables_{table_names}_{timestamp}.sql`
- 压缩文件：在文件名后添加`.gz`

## 导入数据库 (import_database.py)

### 基本用法

#### 1. 导入到新数据库
```bash
# 创建新数据库并导入
python import_database.py --sql-file ./database_export/knowledge_base_full_20241201_143022.sql --create-db

# 指定目标数据库连接参数
python import_database.py --host new-server.com --port 5432 --database new_knowledge_base --user postgres --password new_password --sql-file backup.sql --create-db
```

#### 2. 替换现有数据库
```bash
# 删除现有数据库并重新创建导入
python import_database.py --sql-file backup.sql --drop-existing --create-db
```

#### 3. 导入压缩文件
```bash
python import_database.py --sql-file backup.sql.gz --create-db
```

#### 4. 验证模式（不执行实际导入）
```bash
python import_database.py --sql-file backup.sql --dry-run
```

#### 5. 详细输出模式
```bash
python import_database.py --sql-file backup.sql --create-db --verbose
```

### 导入选项说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 数据库主机地址 | localhost |
| `--port` | 数据库端口 | 5432 |
| `--database` | 目标数据库名称 | knowledge_base |
| `--user` | 数据库用户名 | postgres |
| `--password` | 数据库密码 | postgres |
| `--sql-file` | 要导入的SQL文件路径 | **必需** |
| `--create-db` | 如果数据库不存在则创建 | False |
| `--drop-existing` | 删除现有数据库后重新创建 | False |
| `--dry-run` | 仅验证文件和连接，不执行实际导入 | False |
| `--verbose` | 显示详细输出 | False |

## 完整迁移示例

### 场景1：完整数据库迁移

**步骤1：在源服务器导出数据库**
```bash
# 导出完整数据库并压缩
python export_database.py --host source-server.com --database knowledge_base --user postgres --password source_pass --compress --output-dir /tmp/db_backup
```

**步骤2：传输文件到目标服务器**
```bash
# 使用scp传输文件
scp /tmp/db_backup/knowledge_base_full_20241201_143022.sql.gz user@target-server.com:/tmp/
```

**步骤3：在目标服务器导入数据库**
```bash
# 创建新数据库并导入
python import_database.py --host target-server.com --database knowledge_base --user postgres --password target_pass --sql-file /tmp/knowledge_base_full_20241201_143022.sql.gz --create-db --verbose
```

### 场景2：增量数据迁移

**步骤1：导出特定表的数据**
```bash
# 仅导出新增的数据表
python export_database.py --tables "new_table1,new_table2" --output-dir /tmp/incremental
```

**步骤2：在目标服务器导入**
```bash
# 导入到现有数据库
python import_database.py --sql-file /tmp/incremental/knowledge_base_tables_new_table1_new_table2_20241201_143022.sql
```

### 场景3：结构和数据分离迁移

**步骤1：分别导出结构和数据**
```bash
# 导出结构
python export_database.py --schema-only --output-dir /tmp/migration

# 导出数据
python export_database.py --data-only --compress --output-dir /tmp/migration
```

**步骤2：分别导入结构和数据**
```bash
# 先导入结构
python import_database.py --sql-file /tmp/migration/knowledge_base_schema_20241201_143022.sql --create-db

# 再导入数据
python import_database.py --sql-file /tmp/migration/knowledge_base_data_20241201_143022.sql.gz
```

## 故障排除

### 常见问题

#### 1. 权限错误
```
错误: permission denied for database
```
**解决方案：**
- 确保用户具有足够的数据库权限
- 对于创建/删除数据库，用户需要CREATEDB权限

#### 2. 连接失败
```
错误: could not connect to server
```
**解决方案：**
- 检查主机地址和端口是否正确
- 确保PostgreSQL服务正在运行
- 检查防火墙设置
- 验证pg_hba.conf配置

#### 3. 工具未找到
```
错误: 未找到 pg_dump 工具
```
**解决方案：**
- 安装PostgreSQL客户端工具
- 确保工具在PATH环境变量中

#### 4. 内存不足
对于大型数据库，可能遇到内存问题。
**解决方案：**
- 使用压缩选项减少文件大小
- 分表导出大型数据库
- 增加系统内存或使用更强大的服务器

### 性能优化建议

1. **使用压缩**：对于大型数据库，使用`--compress`选项可以显著减少文件大小
2. **分批处理**：对于超大数据库，考虑分表或分批导出
3. **网络传输**：使用压缩文件进行网络传输以节省带宽
4. **并行处理**：PostgreSQL支持并行导出，可以在pg_dump中使用`-j`参数

## 安全注意事项

1. **密码安全**：
   - 避免在命令行中直接输入密码
   - 使用环境变量或.pgpass文件存储密码
   - 导出完成后及时删除临时文件

2. **文件权限**：
   - 确保导出文件具有适当的权限设置
   - 在传输过程中使用加密连接

3. **数据验证**：
   - 导入后验证数据完整性
   - 比较源数据库和目标数据库的记录数

## 自动化脚本示例

### 定期备份脚本
```bash
#!/bin/bash
# backup_daily.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/postgresql"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
python /path/to/export_database.py \
    --compress \
    --output-dir $BACKUP_DIR

# 删除7天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "备份完成: $BACKUP_DIR"
```

### 迁移验证脚本
```bash
#!/bin/bash
# verify_migration.sh

SOURCE_HOST="source-server.com"
TARGET_HOST="target-server.com"
DATABASE="knowledge_base"

# 比较表数量
SOURCE_TABLES=$(psql -h $SOURCE_HOST -d $DATABASE -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
TARGET_TABLES=$(psql -h $TARGET_HOST -d $DATABASE -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

if [ "$SOURCE_TABLES" -eq "$TARGET_TABLES" ]; then
    echo "表数量验证通过: $SOURCE_TABLES"
else
    echo "表数量不匹配: 源($SOURCE_TABLES) vs 目标($TARGET_TABLES)"
    exit 1
fi

echo "迁移验证完成"
```

通过这些脚本和指南，您可以安全、高效地完成PostgreSQL数据库的迁移工作。