"""
Pydantic models for API request and response
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class JobType(str, Enum):
    """Job type enumeration"""
    MANAGEMENT = "management"
    TECHNICIAN = "technician"
    BLUE_COLLAR = "blue-collar"
    ADMIN = "admin"
    SERVICES = "services"
    RETIRED = "retired"
    ENTREPRENEUR = "entrepreneur"
    STUDENT = "student"
    UNEMPLOYED = "unemployed"
    HOUSEMAID = "housemaid"
    SELF_EMPLOYED = "self-employed"
    UNKNOWN = "unknown"

class MaritalStatus(str, Enum):
    """Marital status enumeration"""
    MARRIED = "married"
    SINGLE = "single"
    DIVORCED = "divorced"

class EducationLevel(str, Enum):
    """Education level enumeration"""
    TERTIARY = "tertiary"
    SECONDARY = "secondary"
    PRIMARY = "primary"
    UNKNOWN = "unknown"

class ContactType(str, Enum):
    """Contact type enumeration"""
    CELLULAR = "cellular"
    TELEPHONE = "telephone"
    UNKNOWN = "unknown"

class MonthType(str, Enum):
    """Month enumeration"""
    JAN = "jan"
    FEB = "feb"
    MAR = "mar"
    APR = "apr"
    MAY = "may"
    JUN = "jun"
    JUL = "jul"
    AUG = "aug"
    SEP = "sep"
    OCT = "oct"
    NOV = "nov"
    DEC = "dec"

class POutcomeType(str, Enum):
    """Previous outcome enumeration"""
    SUCCESS = "success"
    FAILURE = "failure"
    OTHER = "other"
    UNKNOWN = "unknown"

class CustomerRequest(BaseModel):
    """Customer data request model"""
    
    # Required fields
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    job: JobType = Field(..., description="Customer job type")
    marital: MaritalStatus = Field(..., description="Marital status")
    education: EducationLevel = Field(..., description="Education level")
    balance: float = Field(..., description="Annual balance")
    housing: str = Field(..., pattern="^(yes|no)$", description="Has housing loan?")
    loan: str = Field(..., pattern="^(yes|no)$", description="Has personal loan?")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    
    # Optional fields with defaults
    default: str = Field("no", pattern="^(yes|no|unknown)$", description="Has credit default?")
    contact: ContactType = Field(ContactType.CELLULAR, description="Contact communication type")
    day: int = Field(15, ge=1, le=31, description="Last contact day of month")
    month: MonthType = Field(MonthType.MAY, description="Last contact month")
    campaign: int = Field(1, ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(-1, description="Days since last contact (-1 means not contacted)")
    previous: int = Field(0, ge=0, description="Number of contacts before this campaign")
    poutcome: POutcomeType = Field(POutcomeType.UNKNOWN, description="Outcome of previous campaign")
    
    @validator('age')
    def validate_age(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @validator('balance')
    def validate_balance(cls, v):
        if v < -10000 or v > 1000000:
            raise ValueError('Balance must be between -10000 and 1000000')
        return v
    
    @validator('duration')
    def validate_duration(cls, v):
        if v < 0 or v > 5000:
            raise ValueError('Duration must be between 0 and 5000 seconds')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "balance": 1000,
                "housing": "yes",
                "loan": "no",
                "duration": 200,
                "default": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }

class SinglePredictionResponse(BaseModel):
    """Single prediction response model"""
    prediction: str = Field(..., description="Prediction (Yes/No)")
    probability_yes: float = Field(..., description="Probability of subscribing")
    probability_no: float = Field(..., description="Probability of not subscribing")
    confidence_level: str = Field(..., description="Confidence level (High/Medium/Low)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Yes",
                "probability_yes": 0.65,
                "probability_no": 0.35,
                "confidence_level": "Medium"
            }
        }

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    total_customers: int = Field(..., description="Total number of customers")
    predictions: List[dict] = Field(..., description="List of predictions")
    summary: dict = Field(..., description="Summary statistics")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Is model loaded?")
    timestamp: str = Field(..., description="Current timestamp")
