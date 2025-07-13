#!/usr/bin/env python3
"""
Database Models for SutazAI Application
Defines SQLAlchemy ORM models for all core entities.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, 
    ForeignKey, JSON, Float, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from enum import Enum
import uuid
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentType(str, Enum):
    """Agent type enumeration"""
    AUTO_GPT = "auto_gpt"
    LOCAL_AGI = "local_agi"
    AUTO_GEN = "auto_gen"
    BIG_AGI = "big_agi"
    AGENT_ZERO = "agent_zero"
    BROWSER_USE = "browser_use"
    SKYVERN = "skyvern"
    OPEN_WEBUI = "open_webui"
    TABBY_ML = "tabby_ml"
    SEMGREP = "semgrep"
    DOCUMENT_PROCESSOR = "document_processor"
    CODE_GENERATOR = "code_generator"
    SUPREME_AI = "supreme_ai"


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        return pwd_context.verify(password, self.hashed_password)

    def set_password(self, password: str) -> None:
        """Set user password with hashing"""
        self.hashed_password = pwd_context.hash(password)

    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def increment_failed_login(self) -> None:
        """Increment failed login attempts and lock if necessary"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            # Lock for 30 minutes after 5 failed attempts
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)

    def reset_failed_login(self) -> None:
        """Reset failed login attempts"""
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login = datetime.now(timezone.utc)


class UserSession(Base):
    """User session model for tracking active sessions"""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="sessions")


class Agent(Base):
    """AI Agent model"""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(SQLEnum(AgentStatus), default=AgentStatus.INACTIVE)
    version = Column(String(20), nullable=True)
    capabilities = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    health_score = Column(Float, default=1.0)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    total_tasks_completed = Column(Integer, default=0)
    total_tasks_failed = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    tasks = relationship("Task", back_populates="agent", cascade="all, delete-orphan")
    metrics = relationship("AgentMetric", back_populates="agent", cascade="all, delete-orphan")

    def update_health_score(self) -> None:
        """Update agent health score based on performance metrics"""
        if self.total_tasks_completed == 0:
            self.health_score = 1.0
            return
        
        success_rate = self.total_tasks_completed / (self.total_tasks_completed + self.total_tasks_failed)
        response_penalty = min(self.average_response_time / 10.0, 0.5)  # Max 50% penalty
        self.health_score = max(success_rate - response_penalty, 0.0)

    def record_task_completion(self, success: bool, response_time: float) -> None:
        """Record task completion metrics"""
        if success:
            self.total_tasks_completed += 1
        else:
            self.total_tasks_failed += 1
        
        # Update average response time
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        self.average_response_time = ((self.average_response_time * (total_tasks - 1)) + response_time) / total_tasks
        
        self.update_health_score()
        self.last_heartbeat = datetime.now(timezone.utc)


class Task(Base):
    """Task model for tracking AI operations"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(50), nullable=False)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    priority = Column(Integer, default=5)  # 1-10 scale
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    task_metadata = Column(JSON, default=dict)
    progress_percentage = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    estimated_duration = Column(Float, nullable=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="tasks")
    agent = relationship("Agent", back_populates="tasks")
    parent_task = relationship("Task", remote_side=[id], backref="subtasks")
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")

    def update_progress(self, percentage: float, message: str = None) -> None:
        """Update task progress"""
        self.progress_percentage = min(max(percentage, 0.0), 100.0)
        if message:
            self.add_log("PROGRESS", message)

    def mark_started(self) -> None:
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, output_data: dict = None) -> None:
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percentage = 100.0
        if output_data:
            self.output_data = output_data
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()

    def add_log(self, level: str, message: str) -> None:
        """Add a log entry to the task"""
        # This would be called with a session context
        pass


class TaskLog(Base):
    """Task log model for detailed execution tracking"""
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("Task", back_populates="logs")


class Document(Base):
    """Document model for file management"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=True)  # SHA256 hash
    content_extracted = Column(Text, nullable=True)
    task_metadata = Column(JSON, default=dict)
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    user = relationship("User", back_populates="documents")


class AgentMetric(Base):
    """Agent performance metrics"""
    __tablename__ = "agent_metrics"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="metrics")


class SystemConfiguration(Base):
    """System-wide configuration settings"""
    __tablename__ = "system_configurations"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    is_secret = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AuditLog(Base):
    """Audit log for tracking system changes"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(50), nullable=True)
    old_values = Column(JSON, default=dict)
    new_values = Column(JSON, default=dict)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")


# Utility functions for database operations
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, username: str, email: str, password: str, full_name: str = None) -> User:
    """Create a new user"""
    user = User(
        username=username,
        email=email,
        full_name=full_name
    )
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_active_agents(db: Session) -> List[Agent]:
    """Get all active agents"""
    return db.query(Agent).filter(Agent.status == AgentStatus.ACTIVE).all()


def create_task(db: Session, user_id: int, title: str, description: str, task_type: str, **kwargs) -> Task:
    """Create a new task"""
    task = Task(
        user_id=user_id,
        title=title,
        description=description,
        task_type=task_type,
        **kwargs
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task