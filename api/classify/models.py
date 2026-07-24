from __future__ import annotations

import uuid
import datetime as dt

from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.auth.models import User

from core.database import Base


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class ImageUpload(Base):
    __tablename__ = "image_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    blob_key: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="uploads")
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="image_upload",
        cascade="all, delete-orphan",
    )


class ClassificationResponseRecord(Base):
    """One row per Celery task / batch classification request.
    Named *Record to avoid colliding with bentoml_api.schemas.ClassificationResponse (Pydantic)."""
    __tablename__ = "classification_responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    variant: Mapped[str] = mapped_column(String(32), default="default", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="response",
        cascade="all, delete-orphan",
    )


class Prediction(Base):
    """One row per image, per classification response."""
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    response_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("classification_responses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_upload_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    object_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_scores: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    heatmap_blob_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    heatmap_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    response: Mapped["ClassificationResponseRecord"] = relationship("ClassificationResponseRecord", back_populates="predictions")
    image_upload: Mapped["ImageUpload"] = relationship("ImageUpload", back_populates="predictions")