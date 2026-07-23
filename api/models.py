from __future__ import annotations

import uuid
import datetime as dt

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship

from core.database import Base


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    uploads = relationship("ImageUpload", back_populates="user")


class ImageUpload(Base):
    __tablename__ = "image_uploads"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    blob_key = Column(String(512), nullable=False)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(32), default="pending", nullable=False)
    uploaded_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="uploads")
    results = relationship(
        "ClassificationResult",
        back_populates="image_upload",
        cascade="all, delete-orphan",
    )


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_upload_id = Column(
        String(36),
        ForeignKey("image_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id = Column(String(64), nullable=False, unique=True, index=True)

    object_type = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_scores = Column(JSON, nullable=True)
    heatmap_blob_key = Column(String(512), nullable=True)
    heatmap_url = Column(String(1024), nullable=True)

    processing_time_ms = Column(Integer, nullable=True)
    variant = Column(String(32), default="default", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    image_upload = relationship("ImageUpload", back_populates="results")