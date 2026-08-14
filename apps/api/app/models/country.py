from sqlalchemy import Column, String, Integer, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.database.base import Base

class Country(Base):
    __tablename__ = "countries"

    iso3 = Column(String(3), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    region = Column(String(100), nullable=True, index=True)
    income_group = Column(String(100), nullable=True)

    metadata_info = relationship("CountryMetadata", back_populates="country", uselist=False, lazy="selectin")
    historical_indicators = relationship("HistoricalIndicator", back_populates="country", lazy="noload")

class CountryMetadata(Base):
    __tablename__ = "countries_metadata"

    iso3 = Column(String(3), ForeignKey("countries.iso3", ondelete="CASCADE"), primary_key=True)
    capital = Column(String(255), nullable=True)
    population = Column(Integer, nullable=True)
    currency = Column(String(50), nullable=True)
    flag_url = Column(String(1024), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    country = relationship("Country", back_populates="metadata_info")

class HistoricalIndicator(Base):
    __tablename__ = "historical_indicators"
    __table_args__ = (
        Index("ix_hist_ind_iso3_name_year", "iso3", "indicator_name", "year"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    iso3 = Column(String(3), ForeignKey("countries.iso3", ondelete="CASCADE"), index=True)
    indicator_name = Column(String(100), index=True)  # e.g. GDP, Inflation
    year = Column(Integer, index=True)
    value = Column(Float, nullable=False)
    source = Column(String(255), nullable=True)

    country = relationship("Country", back_populates="historical_indicators")
