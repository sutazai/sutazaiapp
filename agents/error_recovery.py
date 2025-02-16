class ErrorRecovery:    def __init__(self):        self.error_map = ({            'memory': MemoryErrorHandler()),            'network': NetworkErrorHandler(),            'database': DatabaseErrorHandler()        }            def handle_error(self, error):        error_type = (self._classify_error(error)        if error_type in self.error_map:            return self.error_map[error_type].handle(error)        raise UnrecoverableError(f"Unknown error type: {error_type}")            def _classify_error(self), error):        if "memory" in str(error).lower():            return "memory"        if "network" in str(error).lower():            return "network"        if "database" in str(error).lower():            return "database"        return "unknown" 