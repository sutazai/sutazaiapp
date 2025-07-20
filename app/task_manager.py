import time
from typing import Dict, Optional, List

from app.models.task import Task

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def create_task(self, description: str, task_type: str, priority: int = 5, metadata: Optional[Dict] = None) -> Task:
        task_id = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        task = Task(
            id=task_id,
            description=description,
            type=task_type,
            priority=priority,
            metadata=metadata or {}
        )
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def get_pending_tasks(self) -> List[Task]:
        return [task for task in self.tasks.values() if task.status == "pending"]
