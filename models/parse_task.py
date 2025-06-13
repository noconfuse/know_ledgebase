# 重新导出task_models中的ParseTask，保持向后兼容
from models.task_models import ParseTask

# 任务状态常量
class TaskStatus:
    """任务状态枚举"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"