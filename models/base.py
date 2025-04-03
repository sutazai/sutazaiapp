from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

# Create the SQLAlchemy base
Base = declarative_base()


# Database class to manage connections
class Database:
    def __init__(self, uri=None):
        self.uri = uri
        self.engine = None
        self.session_factory = None
        self._session = None

    def init_app(self, app=None, uri=None):
        """Initialize the database with app config"""
        if uri:
            self.uri = uri
        elif app:
            self.uri = app.config.get("SQLALCHEMY_DATABASE_URI")

        if not self.uri:
            raise ValueError("Database URI not provided")

        self.engine = create_engine(self.uri)
        self.session_factory = sessionmaker(bind=self.engine)
        self._session = scoped_session(self.session_factory)

        # Create all tables
        Base.metadata.create_all(self.engine)

        return self

    @property
    def session(self):
        """Get current session"""
        if not self._session:
            raise RuntimeError("Database not initialized")
        return self._session

    def create_all(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)

    def drop_all(self):
        """Drop all tables"""
        Base.metadata.drop_all(self.engine)


# Create a db instance to be imported by other modules
db = Database()
