from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379/0')

@celery.task
def heavy_task(data):
    # Perform heavy processing here
    return result 