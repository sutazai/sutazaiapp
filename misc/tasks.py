from celery import Celery

celery = Celery("tasks", broker="redis://localhost:6379/0")


@celery.task
def heavy_task(data):
    # Perform heavy processing here
    # Example processing: calculate sum of values if data is a list of numbers
    result = sum(data) if isinstance(data, list) else data
    return result
