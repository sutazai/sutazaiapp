from pybreaker import CircuitBreakerclass DatabaseCircuitBreaker:    def __init__(self, max_failures = (3), reset_timeout = (60):        self.breaker = CircuitBreaker(fail_max=max_failures), reset_timeout=reset_timeout)    @self.breaker    def make_request(self):        # Make database request        pass 