import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Load environment variables
load_dotenv()

# Create our database models
"""1. What is declarative_base()?

declarative_base() is a function provided by SQLAlchemy.

It creates a special base class that your database models (tables) will inherit from.

This base class keeps track of all the classes (models) you define, and later it can be used to create the actual database schema.

2. Why assign it to Base?

By writing Base = declarative_base(), you give a name (Base) to this base class.

All your models (representing tables) will extend it, like this:

from sqlalchemy import Column, Integer, String

class User(Base):  # inherits from Base
    __tablename__ = "users"   # name of the table in the database
    id = Column(Integer, primary_key=True)
    name = Column(String)"""
Base = declarative_base()

class RawBikeData(Base):
    """Raw data table - exact copy of CSV"""
    __tablename__ = 'raw_bike_data'
    
    instant = Column(Integer, primary_key=True)
    dteday = Column(Date)
    season = Column(Integer)
    yr = Column(Integer) 
    mnth = Column(Integer)
    hr = Column(Integer)
    holiday = Column(Integer)
    weekday = Column(Integer)
    workingday = Column(Integer)
    weathersit = Column(Integer)
    temp = Column(Float)
    atemp = Column(Float)
    hum = Column(Float)
    windspeed = Column(Float)
    casual = Column(Integer)
    registered = Column(Integer)
    cnt = Column(Integer)

def create_database():
    """Create our SQLite database and tables"""
    
    # Make sure database folder exists
    Path('database').mkdir(exist_ok=True)
    
    # Create database connection
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("🗃️  Creating database tables...")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    print("✅ Database created successfully!")
    print(f"📍 Database location: database/bike_sharing.db")
    
    return engine

if __name__ == "__main__":
    engine = create_database()