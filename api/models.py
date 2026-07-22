from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))

class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_b64: Mapped[str] = mapped_column(String, nullable=False)
    variant: Mapped[str] = mapped_column(String(50), nullable=False)
    result: Mapped[str] = mapped_column(String, nullable=True)

