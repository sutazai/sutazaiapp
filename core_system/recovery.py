def recover_from_error(error):
    # Attempt recovery    try:        cleanup_resources()
    # restore_backup()        restart_services()    except Exception as e:
    # print(f" Recovery failed: {e}")        raise
    print(f" Error detected: {error}")
