# Synthetic Data Generator

A powerful web application for generating synthetic datasets with various statistical distributions. Built with Streamlit and Python.

## Features

- Generate synthetic data with multiple distribution types:
  - Normal Distribution
  - Poisson Distribution
  - Uniform Distribution
  - Triangular Distribution
- Interactive column management (add, rename, delete)
- Adjustable dataset length with automatic padding/truncation
- Real-time data preview and statistics
- Distribution visualization
- Export to CSV functionality

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the application:
   - Set the target dataset length in the sidebar
   - Add columns by specifying:
     - Column name
     - Distribution type
     - Distribution parameters
     - Data type
   - Manage columns (rename/delete) using the column management section
   - View data preview, statistics, and distribution plots
   - Save the generated dataset to CSV

## Distribution Parameters

### Normal Distribution
- Mean: Center of the distribution
- Standard Deviation: Spread of the distribution

### Poisson Distribution
- Lambda: Average number of events in the interval

### Uniform Distribution
- Lower Bound: Minimum value
- Upper Bound: Maximum value

### Triangular Distribution
- Left Bound: Minimum value
- Mode: Most likely value
- Right Bound: Maximum value

## License

MIT License 