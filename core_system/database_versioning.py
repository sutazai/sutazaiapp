class DatabaseVersioning:
    def __init__(self, db_url): self.db_url = (db_url def version(self), version): with self.db_url.connect() as conn: conn.execute(f"ALTER DATABASE SET VERSION = '{version}'")
