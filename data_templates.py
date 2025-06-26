import streamlit as st
import pandas as pd
import numpy as np
from data_generator import DataGenerator
from typing import Dict, List, Any

class DataTemplate:
    def __init__(self, name: str, description: str, use_cases: List[str], columns: List[Dict]):
        self.name = name
        self.description = description
        self.use_cases = use_cases
        self.columns = columns

# Pre-defined data templates
TEMPLATES = {
    "financial_stocks": DataTemplate(
        name="📈 Financial Stock Data",
        description="Realistic stock market data with price movements, volume, and market indicators",
        use_cases=[
            "Algorithmic trading strategy development",
            "Portfolio risk analysis",
            "Market trend prediction models",
            "Financial data visualization",
            "Backtesting trading strategies"
        ],
        columns=[
            {
                "name": "Date",
                "type": "DateTime",
                "description": "Trading dates",
                "params": {
                    "frequency": "Daily",
                    "format": "YYYY-MM-DD",
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31"
                }
            },
            {
                "name": "Open_Price",
                "type": "Numeric",
                "description": "Opening stock price",
                "params": {
                    "distribution": "Normal",
                    "mean": 100.0,
                    "std": 20.0,
                    "min_value": 10.0,
                    "max_value": 500.0
                }
            },
            {
                "name": "High_Price",
                "type": "Numeric",
                "description": "Highest price during the day",
                "params": {
                    "distribution": "Normal",
                    "mean": 105.0,
                    "std": 22.0,
                    "min_value": 15.0,
                    "max_value": 550.0
                }
            },
            {
                "name": "Low_Price",
                "type": "Numeric",
                "description": "Lowest price during the day",
                "params": {
                    "distribution": "Normal",
                    "mean": 95.0,
                    "std": 18.0,
                    "min_value": 5.0,
                    "max_value": 450.0
                }
            },
            {
                "name": "Close_Price",
                "type": "Numeric",
                "description": "Closing stock price",
                "params": {
                    "distribution": "Normal",
                    "mean": 102.0,
                    "std": 21.0,
                    "min_value": 12.0,
                    "max_value": 520.0
                }
            },
            {
                "name": "Volume",
                "type": "Numeric",
                "description": "Number of shares traded",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 1000000,
                    "min_value": 10000,
                    "max_value": 10000000
                }
            }
        ]
    ),
    
    "ecommerce_sales": DataTemplate(
        name="🛒 E-commerce Sales Data",
        description="Online retail data with sales, customer information, and product details",
        use_cases=[
            "Customer behavior analysis",
            "Sales forecasting",
            "Inventory management",
            "Marketing campaign analysis",
            "Customer segmentation"
        ],
        columns=[
            {
                "name": "Order_Date",
                "type": "DateTime",
                "description": "Date of order placement",
                "params": {
                    "frequency": "Random",
                    "format": "YYYY-MM-DD",
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31"
                }
            },
            {
                "name": "Customer_ID",
                "type": "Categorical",
                "description": "Unique customer identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"CUST_{i:06d}" for i in range(1, 1001)],
                    "uniform": True
                }
            },
            {
                "name": "Product_Category",
                "type": "Categorical",
                "description": "Product category",
                "params": {
                    "distribution": "custom",
                    "values": ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty", "Toys", "Food"],
                    "uniform": True
                }
            },
            {
                "name": "Product_Name",
                "type": "Categorical",
                "description": "Product name",
                "params": {
                    "distribution": "custom",
                    "values": [f"Product_{i:04d}" for i in range(1, 501)],
                    "uniform": True
                }
            },
            {
                "name": "Quantity",
                "type": "Numeric",
                "description": "Quantity ordered",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 2.0,
                    "min_value": 1,
                    "max_value": 20
                }
            },
            {
                "name": "Unit_Price",
                "type": "Numeric",
                "description": "Price per unit",
                "params": {
                    "distribution": "Uniform",
                    "low": 10.0,
                    "high": 500.0
                }
            },
            {
                "name": "Total_Amount",
                "type": "Numeric",
                "description": "Total order amount",
                "params": {
                    "distribution": "Normal",
                    "mean": 150.0,
                    "std": 100.0,
                    "min_value": 10.0,
                    "max_value": 2000.0
                }
            },
            {
                "name": "Payment_Method",
                "type": "Categorical",
                "description": "Payment method used",
                "params": {
                    "distribution": "custom",
                    "values": ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash on Delivery"],
                    "uniform": True
                }
            }
        ]
    ),
    
    "iot_sensors": DataTemplate(
        name="🌡️ IoT Sensor Data",
        description="Internet of Things sensor data with temperature, humidity, and environmental readings",
        use_cases=[
            "Environmental monitoring",
            "Predictive maintenance",
            "Smart home automation",
            "Industrial IoT applications",
            "Weather pattern analysis"
        ],
        columns=[
            {
                "name": "Timestamp",
                "type": "DateTime",
                "description": "Sensor reading timestamp",
                "params": {
                    "frequency": "Hourly",
                    "format": "YYYY-MM-DD HH:MM:SS",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            },
            {
                "name": "Sensor_ID",
                "type": "Categorical",
                "description": "Unique sensor identifier",
                "params": {
                    "distribution": "custom",
                    "values": ["SENSOR_001", "SENSOR_002", "SENSOR_003", "SENSOR_004", "SENSOR_005"],
                    "uniform": True
                }
            },
            {
                "name": "Temperature",
                "type": "DateTime",
                "description": "Temperature readings with daily patterns",
                "params": {
                    "frequency": "Hourly",
                    "format": "Time Series with Values",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "level": 20.0,
                    "trend": 0.001,
                    "seasonality": 5.0,
                    "seasonality_period": 24,
                    "noise_level": 0.1,
                    "ts_min_value": -10.0,
                    "ts_max_value": 40.0
                }
            },
            {
                "name": "Humidity",
                "type": "DateTime",
                "description": "Humidity readings with seasonal patterns",
                "params": {
                    "frequency": "Hourly",
                    "format": "Time Series with Values",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "level": 60.0,
                    "trend": -0.002,
                    "seasonality": 10.0,
                    "seasonality_period": 24,
                    "noise_level": 0.15,
                    "ts_min_value": 20.0,
                    "ts_max_value": 95.0
                }
            },
            {
                "name": "Pressure",
                "type": "Numeric",
                "description": "Atmospheric pressure",
                "params": {
                    "distribution": "Normal",
                    "mean": 1013.25,
                    "std": 10.0,
                    "min_value": 980.0,
                    "max_value": 1050.0
                }
            },
            {
                "name": "Battery_Level",
                "type": "Numeric",
                "description": "Sensor battery level",
                "params": {
                    "distribution": "Uniform",
                    "low": 10.0,
                    "high": 100.0
                }
            }
        ]
    ),
    
    "healthcare_patients": DataTemplate(
        name="🏥 Healthcare Patient Data",
        description="Medical patient data with demographics, vitals, and treatment information",
        use_cases=[
            "Patient outcome prediction",
            "Disease pattern analysis",
            "Treatment effectiveness studies",
            "Healthcare resource planning",
            "Clinical trial simulation"
        ],
        columns=[
            {
                "name": "Patient_ID",
                "type": "Categorical",
                "description": "Unique patient identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"PAT_{i:06d}" for i in range(1, 1001)],
                    "uniform": True
                }
            },
            {
                "name": "Age",
                "type": "Numeric",
                "description": "Patient age",
                "params": {
                    "distribution": "Normal",
                    "mean": 45.0,
                    "std": 15.0,
                    "min_value": 18,
                    "max_value": 90
                }
            },
            {
                "name": "Gender",
                "type": "Categorical",
                "description": "Patient gender",
                "params": {
                    "distribution": "custom",
                    "values": ["Male", "Female", "Other"],
                    "uniform": False,
                    "probabilities": [0.48, 0.48, 0.04]
                }
            },
            {
                "name": "Blood_Type",
                "type": "Categorical",
                "description": "Blood type",
                "params": {
                    "distribution": "custom",
                    "values": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                    "uniform": False,
                    "probabilities": [0.30, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.11]
                }
            },
            {
                "name": "Heart_Rate",
                "type": "Numeric",
                "description": "Heart rate (bpm)",
                "params": {
                    "distribution": "Normal",
                    "mean": 75.0,
                    "std": 12.0,
                    "min_value": 40,
                    "max_value": 120
                }
            },
            {
                "name": "Blood_Pressure_Systolic",
                "type": "Numeric",
                "description": "Systolic blood pressure",
                "params": {
                    "distribution": "Normal",
                    "mean": 120.0,
                    "std": 15.0,
                    "min_value": 80,
                    "max_value": 180
                }
            },
            {
                "name": "Blood_Pressure_Diastolic",
                "type": "Numeric",
                "description": "Diastolic blood pressure",
                "params": {
                    "distribution": "Normal",
                    "mean": 80.0,
                    "std": 10.0,
                    "min_value": 50,
                    "max_value": 120
                }
            },
            {
                "name": "Temperature",
                "type": "Numeric",
                "description": "Body temperature",
                "params": {
                    "distribution": "Normal",
                    "mean": 37.0,
                    "std": 0.5,
                    "min_value": 35.0,
                    "max_value": 40.0
                }
            },
            {
                "name": "Diagnosis",
                "type": "Categorical",
                "description": "Primary diagnosis",
                "params": {
                    "distribution": "custom",
                    "values": ["Hypertension", "Diabetes", "Asthma", "Heart Disease", "Healthy", "Other"],
                    "uniform": False,
                    "probabilities": [0.25, 0.20, 0.15, 0.10, 0.20, 0.10]
                }
            }
        ]
    ),
    
    "social_media": DataTemplate(
        name="📱 Social Media Analytics",
        description="Social media engagement data with posts, likes, shares, and user interactions",
        use_cases=[
            "Content performance analysis",
            "Audience engagement optimization",
            "Viral content prediction",
            "Social media marketing",
            "User behavior analysis"
        ],
        columns=[
            {
                "name": "Post_Date",
                "type": "DateTime",
                "description": "Date of post",
                "params": {
                    "frequency": "Random",
                    "format": "YYYY-MM-DD",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            },
            {
                "name": "User_ID",
                "type": "Categorical",
                "description": "User identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"USER_{i:06d}" for i in range(1, 501)],
                    "uniform": True
                }
            },
            {
                "name": "Content_Type",
                "type": "Categorical",
                "description": "Type of content posted",
                "params": {
                    "distribution": "custom",
                    "values": ["Image", "Video", "Text", "Story", "Reel"],
                    "uniform": False,
                    "probabilities": [0.40, 0.25, 0.20, 0.10, 0.05]
                }
            },
            {
                "name": "Likes",
                "type": "Numeric",
                "description": "Number of likes",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 50.0,
                    "min_value": 0,
                    "max_value": 1000
                }
            },
            {
                "name": "Shares",
                "type": "Numeric",
                "description": "Number of shares",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 10.0,
                    "min_value": 0,
                    "max_value": 200
                }
            },
            {
                "name": "Comments",
                "type": "Numeric",
                "description": "Number of comments",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 15.0,
                    "min_value": 0,
                    "max_value": 300
                }
            },
            {
                "name": "Reach",
                "type": "Numeric",
                "description": "Number of people who saw the post",
                "params": {
                    "distribution": "Normal",
                    "mean": 1000.0,
                    "std": 500.0,
                    "min_value": 100,
                    "max_value": 5000
                }
            },
            {
                "name": "Engagement_Rate",
                "type": "Numeric",
                "description": "Engagement rate percentage",
                "params": {
                    "distribution": "Normal",
                    "mean": 3.5,
                    "std": 1.5,
                    "min_value": 0.1,
                    "max_value": 10.0
                }
            }
        ]
    ),
    
    "health_metrics": DataTemplate(
        name="🏥 Health Metrics Data",
        description="Health and fitness data with calculated relationships like BMI and health categories",
        use_cases=[
            "Health analytics and research",
            "Fitness app development",
            "Medical data analysis",
            "BMI and health risk assessment",
            "Population health studies"
        ],
        columns=[
            {
                "name": "Patient_ID",
                "type": "Categorical",
                "description": "Unique patient identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"PAT_{i:06d}" for i in range(1, 1001)],
                    "uniform": True
                }
            },
            {
                "name": "Age",
                "type": "Numeric",
                "description": "Patient age in years",
                "params": {
                    "distribution": "Normal",
                    "mean": 45.0,
                    "std": 15.0,
                    "min_value": 18,
                    "max_value": 85
                }
            },
            {
                "name": "Gender",
                "type": "Categorical",
                "description": "Patient gender",
                "params": {
                    "distribution": "custom",
                    "values": ["Male", "Female", "Other"],
                    "uniform": True
                }
            },
            {
                "name": "Height_cm",
                "type": "Numeric",
                "description": "Height in centimeters",
                "params": {
                    "distribution": "Normal",
                    "mean": 170.0,
                    "std": 10.0,
                    "min_value": 140,
                    "max_value": 200
                }
            },
            {
                "name": "Weight_kg",
                "type": "Numeric",
                "description": "Weight in kilograms",
                "params": {
                    "distribution": "Normal",
                    "mean": 70.0,
                    "std": 15.0,
                    "min_value": 40,
                    "max_value": 150
                }
            },
            {
                "name": "BMI",
                "type": "Numeric",
                "description": "Body Mass Index (calculated from height and weight)",
                "params": {
                    "distribution": "calculated",
                    "formula": "Weight_kg / ((Height_cm / 100) ** 2)"
                }
            },
            {
                "name": "BMI_Category",
                "type": "Categorical",
                "description": "BMI health category",
                "params": {
                    "distribution": "conditional",
                    "source_column": "BMI",
                    "thresholds": [18.5, 25, 30],
                    "categories": ["Underweight", "Normal", "Overweight", "Obese"]
                }
            },
            {
                "name": "Blood_Pressure_Systolic",
                "type": "Numeric",
                "description": "Systolic blood pressure",
                "params": {
                    "distribution": "Normal",
                    "mean": 120.0,
                    "std": 15.0,
                    "min_value": 90,
                    "max_value": 180
                }
            },
            {
                "name": "Blood_Pressure_Diastolic",
                "type": "Numeric",
                "description": "Diastolic blood pressure",
                "params": {
                    "distribution": "Normal",
                    "mean": 80.0,
                    "std": 10.0,
                    "min_value": 60,
                    "max_value": 120
                }
            },
            {
                "name": "BP_Category",
                "type": "Categorical",
                "description": "Blood pressure category",
                "params": {
                    "distribution": "conditional",
                    "source_column": "Blood_Pressure_Systolic",
                    "thresholds": [120, 130, 140],
                    "categories": ["Normal", "Elevated", "High Stage 1", "High Stage 2"]
                }
            },
            {
                "name": "Heart_Rate",
                "type": "Numeric",
                "description": "Resting heart rate",
                "params": {
                    "distribution": "Normal",
                    "mean": 72.0,
                    "std": 12.0,
                    "min_value": 50,
                    "max_value": 100
                }
            },
            {
                "name": "Activity_Level",
                "type": "Categorical",
                "description": "Physical activity level",
                "params": {
                    "distribution": "custom",
                    "values": ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                    "uniform": True
                }
            },
            {
                "name": "Calories_Burned",
                "type": "Numeric",
                "description": "Daily calories burned",
                "params": {
                    "distribution": "calculated",
                    "formula": "Weight_kg * 15 + (Heart_Rate - 60) * 2"
                }
            }
        ]
    ),
    
    "student_performance": DataTemplate(
        name="🎓 Student Performance Data",
        description="Academic data with calculated grades, performance categories, and study metrics",
        use_cases=[
            "Educational analytics",
            "Student performance prediction",
            "Academic research",
            "Learning management systems",
            "Educational policy analysis"
        ],
        columns=[
            {
                "name": "Student_ID",
                "type": "Categorical",
                "description": "Unique student identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"STUD_{i:06d}" for i in range(1, 501)],
                    "uniform": True
                }
            },
            {
                "name": "Age",
                "type": "Numeric",
                "description": "Student age",
                "params": {
                    "distribution": "Normal",
                    "mean": 16.0,
                    "std": 2.0,
                    "min_value": 14,
                    "max_value": 20
                }
            },
            {
                "name": "Study_Hours",
                "type": "Numeric",
                "description": "Hours spent studying per week",
                "params": {
                    "distribution": "Normal",
                    "mean": 15.0,
                    "std": 5.0,
                    "min_value": 0,
                    "max_value": 40
                }
            },
            {
                "name": "Sleep_Hours",
                "type": "Numeric",
                "description": "Hours of sleep per night",
                "params": {
                    "distribution": "Normal",
                    "mean": 7.5,
                    "std": 1.5,
                    "min_value": 4,
                    "max_value": 12
                }
            },
            {
                "name": "Attendance_Rate",
                "type": "Numeric",
                "description": "Class attendance percentage",
                "params": {
                    "distribution": "Normal",
                    "mean": 85.0,
                    "std": 10.0,
                    "min_value": 50,
                    "max_value": 100
                }
            },
            {
                "name": "Homework_Score",
                "type": "Numeric",
                "description": "Average homework score",
                "params": {
                    "distribution": "Normal",
                    "mean": 75.0,
                    "std": 15.0,
                    "min_value": 0,
                    "max_value": 100
                }
            },
            {
                "name": "Midterm_Score",
                "type": "Numeric",
                "description": "Midterm exam score",
                "params": {
                    "distribution": "Normal",
                    "mean": 70.0,
                    "std": 20.0,
                    "min_value": 0,
                    "max_value": 100
                }
            },
            {
                "name": "Final_Score",
                "type": "Numeric",
                "description": "Final exam score",
                "params": {
                    "distribution": "Normal",
                    "mean": 72.0,
                    "std": 18.0,
                    "min_value": 0,
                    "max_value": 100
                }
            },
            {
                "name": "Overall_Grade",
                "type": "Numeric",
                "description": "Overall course grade (weighted average)",
                "params": {
                    "distribution": "calculated",
                    "formula": "Homework_Score * 0.3 + Midterm_Score * 0.3 + Final_Score * 0.4"
                }
            },
            {
                "name": "Grade_Letter",
                "type": "Categorical",
                "description": "Letter grade based on overall score",
                "params": {
                    "distribution": "conditional",
                    "source_column": "Overall_Grade",
                    "thresholds": [60, 70, 80, 90],
                    "categories": ["F", "D", "C", "B", "A"]
                }
            },
            {
                "name": "Performance_Level",
                "type": "Categorical",
                "description": "Performance category",
                "params": {
                    "distribution": "conditional",
                    "source_column": "Overall_Grade",
                    "thresholds": [70, 85],
                    "categories": ["Needs Improvement", "Satisfactory", "Excellent"]
                }
            },
            {
                "name": "Study_Efficiency",
                "type": "Numeric",
                "description": "Study efficiency score",
                "params": {
                    "distribution": "calculated",
                    "formula": "Overall_Grade / Study_Hours * 10"
                }
            },
            {
                "name": "Health_Score",
                "type": "Numeric",
                "description": "Overall health and wellness score",
                "params": {
                    "distribution": "calculated",
                    "formula": "Sleep_Hours * 10 + Attendance_Rate * 0.5"
                }
            },
            {
                "name": "Risk_Factor",
                "type": "Categorical",
                "description": "Academic risk assessment",
                "params": {
                    "distribution": "conditional",
                    "source_column": "Overall_Grade",
                    "thresholds": [60, 75],
                    "categories": ["High Risk", "Medium Risk", "Low Risk"]
                }
            }
        ]
    ),
    
    "fintech_payment_app": DataTemplate(
        name="💳 Fintech Payment App Data",
        description="Comprehensive payment app data with transactions, user behavior, fraud patterns, and financial metrics",
        use_cases=[
            "Fraud detection and prevention",
            "User behavior analysis",
            "Transaction pattern recognition",
            "Risk assessment and scoring",
            "Customer segmentation",
            "Revenue optimization",
            "Compliance and regulatory reporting",
            "Product feature analysis",
            "Customer lifetime value modeling",
            "Real-time transaction monitoring"
        ],
        columns=[
            {
                "name": "Transaction_ID",
                "type": "Categorical",
                "description": "Unique transaction identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"TXN_{i:08d}" for i in range(1, 10001)],
                    "uniform": True
                }
            },
            {
                "name": "Timestamp",
                "type": "DateTime",
                "description": "Transaction timestamp with realistic patterns",
                "params": {
                    "frequency": "Random",
                    "format": "YYYY-MM-DD HH:MM:SS",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            },
            {
                "name": "User_ID",
                "type": "Categorical",
                "description": "Unique user identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"USER_{i:06d}" for i in range(1, 5001)],
                    "uniform": True
                }
            },
            {
                "name": "User_Age",
                "type": "Numeric",
                "description": "User age with realistic distribution",
                "params": {
                    "distribution": "Normal",
                    "mean": 35.0,
                    "std": 12.0,
                    "min_value": 18,
                    "max_value": 80
                }
            },
            {
                "name": "User_Location_Country",
                "type": "Categorical",
                "description": "User's country with weighted distribution",
                "params": {
                    "distribution": "custom",
                    "values": ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France", "Japan", "India", "Brazil", "Mexico"],
                    "uniform": False,
                    "probabilities": [0.35, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.05]
                }
            },
            {
                "name": "User_Device_Type",
                "type": "Categorical",
                "description": "Device used for transaction",
                "params": {
                    "distribution": "custom",
                    "values": ["Mobile App", "Web Browser", "Smart Watch", "Tablet"],
                    "uniform": False,
                    "probabilities": [0.65, 0.25, 0.05, 0.05]
                }
            },
            {
                "name": "Transaction_Type",
                "type": "Categorical",
                "description": "Type of payment transaction",
                "params": {
                    "distribution": "custom",
                    "values": ["Peer-to-Peer", "Merchant Payment", "Bill Payment", "Money Transfer", "Subscription", "Investment", "Crypto Purchase", "Gift Card"],
                    "uniform": False,
                    "probabilities": [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
                }
            },
            {
                "name": "Transaction_Amount",
                "type": "Numeric",
                "description": "Transaction amount with realistic distribution",
                "params": {
                    "distribution": "Exponential",
                    "lambda": 0.01,
                    "min_value": 0.01,
                    "max_value": 10000.0
                }
            },
            {
                "name": "Transaction_Currency",
                "type": "Categorical",
                "description": "Transaction currency",
                "params": {
                    "distribution": "custom",
                    "values": ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR", "BRL", "MXN"],
                    "uniform": False,
                    "probabilities": [0.45, 0.20, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]
                }
            },
            {
                "name": "Payment_Method",
                "type": "Categorical",
                "description": "Payment method used",
                "params": {
                    "distribution": "custom",
                    "values": ["Credit Card", "Debit Card", "Bank Transfer", "Digital Wallet", "Crypto", "Gift Card", "Buy Now Pay Later"],
                    "uniform": False,
                    "probabilities": [0.35, 0.25, 0.20, 0.10, 0.03, 0.02, 0.05]
                }
            },
            {
                "name": "Merchant_Category",
                "type": "Categorical",
                "description": "Merchant category code",
                "params": {
                    "distribution": "custom",
                    "values": ["Retail", "Food & Dining", "Transportation", "Entertainment", "Healthcare", "Education", "Travel", "Utilities", "Insurance", "Other"],
                    "uniform": False,
                    "probabilities": [0.25, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
                }
            },
            {
                "name": "Merchant_ID",
                "type": "Categorical",
                "description": "Merchant identifier",
                "params": {
                    "distribution": "custom",
                    "values": [f"MERCH_{i:05d}" for i in range(1, 2001)],
                    "uniform": True
                }
            },
            {
                "name": "Transaction_Status",
                "type": "Categorical",
                "description": "Transaction status",
                "params": {
                    "distribution": "custom",
                    "values": ["Completed", "Pending", "Failed", "Cancelled", "Refunded"],
                    "uniform": False,
                    "probabilities": [0.85, 0.08, 0.04, 0.02, 0.01]
                }
            },
            {
                "name": "Fraud_Score",
                "type": "Numeric",
                "description": "Fraud risk score (0-100)",
                "params": {
                    "distribution": "Normal",
                    "mean": 15.0,
                    "std": 20.0,
                    "min_value": 0.0,
                    "max_value": 100.0
                }
            },
            {
                "name": "Is_Fraudulent",
                "type": "Boolean",
                "description": "Fraud flag based on fraud score",
                "params": {
                    "p": 0.02
                }
            },
            {
                "name": "Transaction_Duration_Seconds",
                "type": "Numeric",
                "description": "Time taken to complete transaction",
                "params": {
                    "distribution": "Exponential",
                    "lambda": 0.5,
                    "min_value": 1.0,
                    "max_value": 300.0
                }
            },
            {
                "name": "User_Session_Duration_Minutes",
                "type": "Numeric",
                "description": "User session duration",
                "params": {
                    "distribution": "Normal",
                    "mean": 8.0,
                    "std": 5.0,
                    "min_value": 0.1,
                    "max_value": 60.0
                }
            },
            {
                "name": "User_Account_Age_Days",
                "type": "Numeric",
                "description": "Days since user account creation",
                "params": {
                    "distribution": "Uniform",
                    "low": 1.0,
                    "high": 3650.0
                }
            },
            {
                "name": "User_Transaction_Count",
                "type": "Numeric",
                "description": "Total transactions by user",
                "params": {
                    "distribution": "Poisson",
                    "lambda": 25.0,
                    "min_value": 1,
                    "max_value": 1000
                }
            },
            {
                "name": "User_Total_Spent",
                "type": "Numeric",
                "description": "Total amount spent by user",
                "params": {
                    "distribution": "Exponential",
                    "lambda": 0.0001,
                    "min_value": 0.01,
                    "max_value": 100000.0
                }
            },
            {
                "name": "User_Avg_Transaction_Amount",
                "type": "Numeric",
                "description": "Average transaction amount per user",
                "params": {
                    "distribution": "Normal",
                    "mean": 85.0,
                    "std": 50.0,
                    "min_value": 1.0,
                    "max_value": 5000.0
                }
            },
            {
                "name": "User_Risk_Score",
                "type": "Numeric",
                "description": "User risk assessment score",
                "params": {
                    "distribution": "Normal",
                    "mean": 25.0,
                    "std": 15.0,
                    "min_value": 0.0,
                    "max_value": 100.0
                }
            },
            {
                "name": "User_Verification_Status",
                "type": "Categorical",
                "description": "User verification level",
                "params": {
                    "distribution": "custom",
                    "values": ["Unverified", "Basic", "Enhanced", "Premium"],
                    "uniform": False,
                    "probabilities": [0.15, 0.45, 0.30, 0.10]
                }
            },
            {
                "name": "Transaction_Hour",
                "type": "Numeric",
                "description": "Hour of transaction (0-23)",
                "params": {
                    "distribution": "Normal",
                    "mean": 14.0,
                    "std": 6.0,
                    "min_value": 0,
                    "max_value": 23
                }
            },
            {
                "name": "Transaction_Day_Of_Week",
                "type": "Categorical",
                "description": "Day of week for transaction",
                "params": {
                    "distribution": "custom",
                    "values": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    "uniform": False,
                    "probabilities": [0.12, 0.13, 0.14, 0.15, 0.16, 0.15, 0.15]
                }
            },
            {
                "name": "Is_Weekend",
                "type": "Boolean",
                "description": "Whether transaction occurred on weekend",
                "params": {
                    "p": 0.30
                }
            },
            {
                "name": "Is_Holiday",
                "type": "Boolean",
                "description": "Whether transaction occurred on holiday",
                "params": {
                    "p": 0.08
                }
            },
            {
                "name": "Transaction_Fee",
                "type": "Numeric",
                "description": "Transaction fee charged",
                "params": {
                    "distribution": "Normal",
                    "mean": 2.50,
                    "std": 1.50,
                    "min_value": 0.0,
                    "max_value": 15.0
                }
            },
            {
                "name": "Exchange_Rate",
                "type": "Numeric",
                "description": "Exchange rate for foreign currency transactions",
                "params": {
                    "distribution": "Normal",
                    "mean": 1.0,
                    "std": 0.2,
                    "min_value": 0.5,
                    "max_value": 2.0
                }
            },
            {
                "name": "Network_Response_Time_MS",
                "type": "Numeric",
                "description": "Network response time in milliseconds",
                "params": {
                    "distribution": "Exponential",
                    "lambda": 0.01,
                    "min_value": 50.0,
                    "max_value": 5000.0
                }
            },
            {
                "name": "App_Version",
                "type": "Categorical",
                "description": "App version used for transaction",
                "params": {
                    "distribution": "custom",
                    "values": ["1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0", "2.2.0"],
                    "uniform": False,
                    "probabilities": [0.05, 0.10, 0.15, 0.25, 0.30, 0.15]
                }
            },
            {
                "name": "User_Notification_Enabled",
                "type": "Boolean",
                "description": "Whether user has notifications enabled",
                "params": {
                    "p": 0.75
                }
            },
            {
                "name": "Biometric_Authentication_Used",
                "type": "Boolean",
                "description": "Whether biometric authentication was used",
                "params": {
                    "p": 0.60
                }
            },
            {
                "name": "Location_Accuracy_Meters",
                "type": "Numeric",
                "description": "GPS location accuracy",
                "params": {
                    "distribution": "Normal",
                    "mean": 50.0,
                    "std": 30.0,
                    "min_value": 5.0,
                    "max_value": 500.0
                }
            },
            {
                "name": "Battery_Level_Percent",
                "type": "Numeric",
                "description": "Device battery level at transaction time",
                "params": {
                    "distribution": "Uniform",
                    "low": 5.0,
                    "high": 100.0
                }
            },
            {
                "name": "Network_Type",
                "type": "Categorical",
                "description": "Network connection type",
                "params": {
                    "distribution": "custom",
                    "values": ["WiFi", "4G", "5G", "3G"],
                    "uniform": False,
                    "probabilities": [0.40, 0.35, 0.20, 0.05]
                }
            },
            {
                "name": "Transaction_Success_Probability",
                "type": "Numeric",
                "description": "Predicted success probability",
                "params": {
                    "distribution": "Normal",
                    "mean": 0.85,
                    "std": 0.15,
                    "min_value": 0.0,
                    "max_value": 1.0
                }
            },
            {
                "name": "Customer_Satisfaction_Score",
                "type": "Numeric",
                "description": "Post-transaction satisfaction score",
                "params": {
                    "distribution": "Normal",
                    "mean": 4.2,
                    "std": 0.8,
                    "min_value": 1.0,
                    "max_value": 5.0
                }
            },
            {
                "name": "Support_Ticket_Created",
                "type": "Boolean",
                "description": "Whether support ticket was created",
                "params": {
                    "p": 0.03
                }
            },
            {
                "name": "Transaction_Dispute_Amount",
                "type": "Numeric",
                "description": "Amount disputed by user",
                "params": {
                    "distribution": "Exponential",
                    "lambda": 0.001,
                    "min_value": 0.0,
                    "max_value": 1000.0
                }
            },
            {
                "name": "Regulatory_Compliance_Score",
                "type": "Numeric",
                "description": "Compliance score for regulatory reporting",
                "params": {
                    "distribution": "Normal",
                    "mean": 95.0,
                    "std": 5.0,
                    "min_value": 70.0,
                    "max_value": 100.0
                }
            }
        ]
    )
}

def show_data_templates_page():
    """Show the data templates page with pre-configured datasets."""
    st.title("📋 Data Templates")
    st.write("Choose from pre-configured templates to quickly generate realistic synthetic datasets for common use cases.")
    
    # Template selection
    st.header("🎯 Select a Template")
    
    template_names = list(TEMPLATES.keys())
    selected_template_key = st.selectbox(
        "Choose a template:",
        options=template_names,
        format_func=lambda x: TEMPLATES[x].name
    )
    
    if selected_template_key:
        template = TEMPLATES[selected_template_key]
        
        # Display template information
        st.header(f"📋 {template.name}")
        st.write(template.description)
        
        # Use cases
        st.subheader("🎯 Use Cases")
        for use_case in template.use_cases:
            st.write(f"• {use_case}")
        
        # Preview columns
        st.subheader("📊 Dataset Structure")
        
        # Create a preview table
        preview_data = []
        for col in template.columns:
            preview_data.append({
                "Column Name": col["name"],
                "Type": col["type"],
                "Description": col["description"],
                "Distribution": get_distribution_info(col)
            })
        
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        
        # Configuration options
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_size = st.number_input(
                "Dataset Size (rows)",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100
            )
        
        with col2:
            if st.button("🚀 Generate Dataset", type="primary", use_container_width=True):
                generate_template_dataset(template, dataset_size)
        
        # Template details
        with st.expander("🔍 Template Details"):
            st.write("**Column Configurations:**")
            for i, col in enumerate(template.columns, 1):
                st.write(f"**{i}. {col['name']}** ({col['type']})")
                st.write(f"   - Description: {col['description']}")
                st.write(f"   - Parameters: {col['params']}")
                st.write("")

def get_distribution_info(col: Dict) -> str:
    """Get a human-readable description of the column distribution."""
    params = col["params"]
    
    if col["type"] == "Numeric":
        dist = params.get("distribution", "Unknown")
        if dist == "Normal":
            return f"Normal (μ={params.get('mean', 'N/A')}, σ={params.get('std', 'N/A')})"
        elif dist == "Uniform":
            return f"Uniform ({params.get('low', 'N/A')} to {params.get('high', 'N/A')})"
        elif dist == "Poisson":
            return f"Poisson (λ={params.get('lambda', 'N/A')})"
        else:
            return dist
    elif col["type"] == "Categorical":
        if params.get("distribution") == "faker":
            return f"Faker ({params.get('provider', 'N/A')}.{params.get('method', 'N/A')})"
        else:
            values = params.get("values", [])
            return f"Custom ({len(values)} values)"
    elif col["type"] == "DateTime":
        if params.get("format") == "Time Series with Values":
            return f"Time Series ({params.get('frequency', 'N/A')})"
        else:
            return f"DateTime ({params.get('frequency', 'N/A')})"
    else:
        return "Unknown"

def generate_template_dataset(template: DataTemplate, size: int):
    """Generate a dataset using the selected template."""
    try:
        # Initialize generator if not exists
        if 'generator' not in st.session_state:
            st.session_state.generator = DataGenerator()
        
        # Set target length
        st.session_state.generator.set_target_length(size)
        
        # Generate each column
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, col in enumerate(template.columns):
            status_text.text(f"Generating {col['name']}...")
            
            # Convert template parameters to generator format
            column_config = convert_template_to_generator(col, size)
            
            # Add column
            st.session_state.generator.add_column(
                name=column_config['name'],
                distribution=column_config['distribution'],
                parameters=column_config['parameters'],
                dtype=column_config['dtype'],
                length=column_config['length'],
                min_value=column_config['min_value'],
                max_value=column_config['max_value'],
                column_type=column_config['column_type']
            )
            
            progress_bar.progress((i + 1) / len(template.columns))
        
        status_text.text("✅ Dataset generated successfully!")
        st.success(f"Generated {template.name} dataset with {size} rows and {len(template.columns)} columns!")
        
        # Show preview
        st.subheader("📊 Generated Data Preview")
        st.dataframe(st.session_state.generator.data.head(10), use_container_width=True)
        
        # Download option
        st.subheader("💾 Download Dataset")
        filename = f"{template.name.lower().replace(' ', '_').replace('📈', '').replace('🛒', '').replace('🌡️', '').replace('🏥', '').replace('📱', '')}_{size}_rows.csv"
        
        if st.button("📥 Download CSV", use_container_width=True):
            st.session_state.generator.save_to_csv(filename)
            st.success(f"Dataset saved as {filename}")
        
    except Exception as e:
        st.error(f"Error generating dataset: {str(e)}")

def convert_template_to_generator(col: Dict, size: int) -> Dict:
    """Convert template column configuration to generator format."""
    column_type = col["type"].lower()
    
    if column_type == "numeric":
        params = col["params"].copy()
        distribution = params.pop("distribution")
        
        return {
            "name": col["name"],
            "distribution": distribution,
            "parameters": params,
            "dtype": "float64",
            "length": size,
            "min_value": params.get("min_value"),
            "max_value": params.get("max_value"),
            "column_type": "numeric"
        }
    
    elif column_type == "categorical":
        if col["params"].get("distribution") == "faker":
            return {
                "name": col["name"],
                "distribution": "faker",
                "parameters": col["params"],
                "dtype": "object",
                "length": size,
                "min_value": None,
                "max_value": None,
                "column_type": "categorical"
            }
        else:
            return {
                "name": col["name"],
                "distribution": "custom",
                "parameters": col["params"],
                "dtype": "object",
                "length": size,
                "min_value": None,
                "max_value": None,
                "column_type": "categorical"
            }
    
    elif column_type == "datetime":
        return {
            "name": col["name"],
            "distribution": "datetime",
            "parameters": col["params"],
            "dtype": "datetime64[ns]",
            "length": size,
            "min_value": None,
            "max_value": None,
            "column_type": "datetime"
        }
    
    elif column_type == "boolean":
        return {
            "name": col["name"],
            "distribution": "Binary",
            "parameters": col["params"],
            "dtype": "bool",
            "length": size,
            "min_value": None,
            "max_value": None,
            "column_type": "boolean"
        }
    
    else:
        raise ValueError(f"Unsupported column type: {column_type}") 