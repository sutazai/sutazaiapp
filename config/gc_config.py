import gc
import threading
import time

def configure_gc():
    """Configure Python garbage collection for optimal memory usage"""
    # Set garbage collection thresholds
    gc.set_threshold(700, 10, 10)
    
    # Enable garbage collection debugging (disabled in production)
    # gc.set_debug(gc.DEBUG_STATS)
    
    # Force garbage collection every 5 minutes
    def periodic_gc():
        while True:
            time.sleep(300)  # 5 minutes
            collected = gc.collect()
            if collected > 0:
                print(f"GC collected {collected} objects")
    
    gc_thread = threading.Thread(target=periodic_gc, daemon=True)
    gc_thread.start()

if __name__ == "__main__":
    configure_gc()
