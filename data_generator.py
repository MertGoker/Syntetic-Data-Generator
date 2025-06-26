import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List, Union, Tuple
import copy
from faker import Faker
import random
from collections import Counter

class DataGenerator:
    def __init__(self):
        self.data = pd.DataFrame()
        self.column_configs = {}
        self.target_length = None
        self.column_relationships = {}
        self.history = []  # List to store historical states
        self.current_history_index = -1  # Index of current state in history
        self.max_history_size = 50  # Maximum number of states to keep
        self.faker = Faker()
        
        # Save initial state
        self._save_state()

    def _save_state(self) -> None:
        """Save current state to history."""
        # Create deep copy of current state
        state = {
            'data': self.data.copy(),
            'column_configs': copy.deepcopy(self.column_configs),
            'column_relationships': copy.deepcopy(self.column_relationships),
            'target_length': self.target_length
        }
        
        # If we're not at the end of history, remove future states
        if self.current_history_index < len(self.history) - 1:
            self.history = self.history[:self.current_history_index + 1]
        
        # Add to history
        self.history.append(state)
        self.current_history_index = len(self.history) - 1
        
        # Trim history if it gets too long
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
            self.current_history_index = len(self.history) - 1

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a state from history."""
        self.data = state['data'].copy()
        self.column_configs = copy.deepcopy(state['column_configs'])
        self.column_relationships = copy.deepcopy(state['column_relationships'])
        self.target_length = state['target_length']

    def undo(self) -> bool:
        """Undo the last change. Returns True if successful, False if no more history."""
        if self.current_history_index > 0 and len(self.history) > 1:
            self.current_history_index -= 1
            self._restore_state(self.history[self.current_history_index])
            return True
        return False

    def redo(self) -> bool:
        """Redo the last undone change. Returns True if successful, False if no more future states."""
        if self.current_history_index < len(self.history) - 1:
            self.current_history_index += 1
            self._restore_state(self.history[self.current_history_index])
            return True
        return False

    def reset_to_initial(self) -> None:
        """Reset to the initial state."""
        if self.history:
            self._restore_state(self.history[0])
            self.current_history_index = 0

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_history_index > 0 and len(self.history) > 1

    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_history_index < len(self.history) - 1

    def set_target_length(self, length: int) -> None:
        """Set the target length for the dataset."""
        self.target_length = length
        self._save_state()

    def _generate_value_within_bounds(self, 
                                    distribution: str, 
                                    parameters: Dict[str, float], 
                                    min_value: Optional[float], 
                                    max_value: Optional[float]) -> float:
        """Generate a single value within specified bounds."""
        while True:
            if distribution == 'Normal':
                value = np.random.normal(
                    loc=parameters.get('mean', 0),
                    scale=parameters.get('std', 1)
                )
            elif distribution == 'Poisson':
                value = np.random.poisson(
                    lam=parameters.get('lambda', 1)
                )
            elif distribution == 'Uniform':
                value = np.random.uniform(
                    low=parameters.get('low', 0),
                    high=parameters.get('high', 1)
                )
            elif distribution == 'Triangular':
                value = np.random.triangular(
                    left=parameters.get('left', 0),
                    mode=parameters.get('mode', 0.5),
                    right=parameters.get('right', 1)
                )
            elif distribution == 'Exponential':
                value = np.random.exponential(
                    scale=1/parameters.get('lambda', 1)
                )
            elif distribution == 'Binary':
                value = np.random.choice([0, 1], p=[1-parameters.get('p', 0.5), parameters.get('p', 0.5)])
            elif distribution == 'calculated':
                # For calculated columns, return a placeholder value
                # The actual calculation will be done by the interrelations module
                value = 0.0
            else:
                raise ValueError(f"Unsupported distribution type: {distribution}")

            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
            return value

    def _check_cardinality(self, num_unique: int, num_rows: int) -> bool:
        """Check if the cardinality is reasonable for the dataset size."""
        # Warn if unique values are more than 10% of total rows
        return num_unique <= (num_rows * 0.1)

    def _generate_categorical_data(self,
                                 num_rows: int,
                                 method: str,
                                 params: Dict[str, Any]) -> np.ndarray:
        """
        Generate categorical data using either Faker or custom value-probability table.
        
        Args:
            num_rows: Number of rows to generate
            method: 'faker' or 'custom'
            params: Parameters for data generation
        """
        if method == 'faker':
            # Get Faker provider and method
            provider = params.get('provider', 'name')
            method = params.get('method', 'full_name')
            
            # Get the Faker method
            faker_method = getattr(getattr(self.faker, provider), method)
            
            # Generate unique values first to check cardinality
            unique_values = set()
            max_attempts = min(num_rows * 2, 10000)  # Limit attempts to prevent infinite loops
            attempts = 0
            
            while len(unique_values) < params.get('num_unique', 10) and attempts < max_attempts:
                unique_values.add(faker_method())
                attempts += 1
            
            # Check cardinality
            if not self._check_cardinality(len(unique_values), num_rows):
                raise ValueError(
                    f"Too many unique values ({len(unique_values)}) for the dataset size ({num_rows}). "
                    f"Consider reducing the number of unique values or increasing the dataset size."
                )
            
            # Generate the actual data
            unique_values = list(unique_values)
            if params.get('uniform', True):
                # Uniform distribution
                return np.random.choice(unique_values, size=num_rows)
            else:
                # Custom probabilities
                probs = params.get('probabilities', [1/len(unique_values)] * len(unique_values))
                if len(probs) != len(unique_values):
                    raise ValueError("Number of probabilities must match number of unique values")
                if not np.isclose(sum(probs), 1.0):
                    raise ValueError("Probabilities must sum to 1")
                return np.random.choice(unique_values, size=num_rows, p=probs)
                
        elif method == 'custom':
            values = params.get('values', [])
            probabilities = params.get('probabilities', None)
            
            if not values:
                raise ValueError("No values provided for custom categorical data")
            
            # Check cardinality
            if not self._check_cardinality(len(values), num_rows):
                raise ValueError(
                    f"Too many unique values ({len(values)}) for the dataset size ({num_rows}). "
                    f"Consider reducing the number of unique values or increasing the dataset size."
                )
            
            if probabilities is None:
                # Uniform distribution
                return np.random.choice(values, size=num_rows)
            else:
                # Custom probabilities
                if len(probabilities) != len(values):
                    raise ValueError("Number of probabilities must match number of values")
                if not np.isclose(sum(probabilities), 1.0):
                    raise ValueError("Probabilities must sum to 1")
                return np.random.choice(values, size=num_rows, p=probabilities)
        else:
            raise ValueError(f"Unsupported categorical generation method: {method}")

    def add_column(self, 
                  name: str, 
                  distribution: str, 
                  parameters: Dict[str, float], 
                  dtype: str = 'float64',
                  length: Optional[int] = None,
                  min_value: Optional[float] = None,
                  max_value: Optional[float] = None,
                  column_type: str = 'numeric') -> None:
        """
        Add a new column to the dataset with specified distribution and parameters.
        
        Args:
            name: Column name
            distribution: Type of distribution or generation method
            parameters: Dictionary of parameters
            dtype: Data type of the column
            length: Length of the column
            min_value: Minimum value for numeric columns
            max_value: Maximum value for numeric columns
            column_type: Type of column ('numeric', 'categorical', 'boolean', 'datetime')
        """
        if length is None and self.target_length is None:
            raise ValueError("Either length or target_length must be specified")

        actual_length = length if length is not None else self.target_length
        
        if column_type == 'categorical':
            try:
                data = self._generate_categorical_data(
                    num_rows=actual_length,
                    method=distribution,  # 'faker' or 'custom'
                    params=parameters
                )
            except ValueError as e:
                raise ValueError(str(e))
        elif column_type == 'boolean':
            data = self._generate_boolean_data(actual_length, parameters)
        elif column_type == 'datetime':
            data = self._generate_datetime_data(actual_length, parameters)
            
            # Handle special case where datetime returns a DataFrame
            if isinstance(data, pd.DataFrame):
                # This is a time series with values - add both columns
                datetime_col = data['datetime']
                values_col = data['value']
                
                # Add datetime column
                self.column_configs[name] = {
                    'distribution': distribution,
                    'parameters': parameters,
                    'dtype': 'datetime64[ns]',
                    'length': actual_length,
                    'min_value': None,
                    'max_value': None,
                    'column_type': 'datetime'
                }
                self.data[name] = datetime_col
                
                # Add values column
                values_name = f"{name}_value"
                self.column_configs[values_name] = {
                    'distribution': 'time_series',
                    'parameters': parameters,
                    'dtype': 'float64',
                    'length': actual_length,
                    'min_value': None,
                    'max_value': None,
                    'column_type': 'numeric'
                }
                self.data[values_name] = values_col
                
                self._save_state()
                return
            else:
                # Regular datetime column
                data = data
        else:  # numeric
            if distribution == 'binary':
                data = np.random.choice([0, 1], size=actual_length, p=[1-parameters.get('p', 0.5), parameters.get('p', 0.5)])
            elif distribution == 'exponential':
                data = np.random.exponential(scale=1/parameters.get('lambda', 1), size=actual_length)
            else:
                # Generate data with bounds checking
                data = np.array([
                    self._generate_value_within_bounds(distribution, parameters, min_value, max_value)
                    for _ in range(actual_length)
                ])

        # Convert to specified dtype
        if dtype == 'datetime64[ns]' and column_type == 'datetime':
            # Handle datetime conversion
            if parameters.get('format') == 'YYYY-MM-DD':
                data = pd.to_datetime(data)
            else:
                data = pd.to_datetime(data)
        else:
            data = data.astype(dtype)
        
        # Store column configuration
        self.column_configs[name] = {
            'distribution': distribution,
            'parameters': parameters,
            'dtype': dtype,
            'length': actual_length,
            'min_value': min_value,
            'max_value': max_value,
            'column_type': column_type
        }
        
        # Add to DataFrame
        self.data[name] = data
        self._save_state()

    def add_column_relationship(self,
                              target_column: str,
                              source_column: str,
                              relationship_type: str,
                              parameters: Dict[str, float]) -> None:
        """
        Add a relationship between columns.
        
        Args:
            target_column: Column to be generated based on relationship
            source_column: Column that influences the target
            relationship_type: Type of relationship ('linear', 'polynomial', 'exponential', 'logarithmic', 'trigonometric', 'conditional')
            parameters: Parameters for the relationship
        """
        if target_column not in self.data.columns or source_column not in self.data.columns:
            raise ValueError("Both columns must exist in the dataset")

        self.column_relationships[target_column] = {
            'source_column': source_column,
            'relationship_type': relationship_type,
            'parameters': parameters
        }

        # Generate new values based on relationship
        source_data = self.data[source_column].values
        
        if relationship_type == 'linear':
            new_data = parameters.get('slope', 1) * source_data + parameters.get('intercept', 0)
        elif relationship_type == 'polynomial':
            new_data = (parameters.get('a', 1) * source_data**2 + 
                       parameters.get('b', 0) * source_data + 
                       parameters.get('c', 0))
        elif relationship_type == 'exponential':
            new_data = parameters.get('a', 1) * np.exp(parameters.get('b', 1) * source_data)
        elif relationship_type == 'logarithmic':
            # Add small constant to avoid log(0)
            safe_data = source_data + parameters.get('offset', 1)
            new_data = parameters.get('a', 1) * np.log(safe_data) + parameters.get('b', 0)
        elif relationship_type == 'trigonometric':
            func_type = parameters.get('function', 'sin')
            if func_type == 'sin':
                new_data = parameters.get('a', 1) * np.sin(parameters.get('b', 1) * source_data) + parameters.get('c', 0)
            elif func_type == 'cos':
                new_data = parameters.get('a', 1) * np.cos(parameters.get('b', 1) * source_data) + parameters.get('c', 0)
            elif func_type == 'tan':
                new_data = parameters.get('a', 1) * np.tan(parameters.get('b', 1) * source_data) + parameters.get('c', 0)
            else:
                raise ValueError(f"Unsupported trigonometric function: {func_type}")
        elif relationship_type == 'conditional':
            # Conditional relationship based on thresholds
            threshold = parameters.get('threshold', 0)
            low_value = parameters.get('low_value', 0)
            high_value = parameters.get('high_value', 1)
            new_data = np.where(source_data > threshold, high_value, low_value)
        elif relationship_type == 'inverse':
            # Inverse relationship
            safe_data = source_data + parameters.get('offset', 1e-8)  # Avoid division by zero
            new_data = parameters.get('a', 1) / safe_data + parameters.get('b', 0)
        else:
            raise ValueError(f"Unsupported relationship type: {relationship_type}")

        # Apply min/max constraints if they exist
        config = self.column_configs[target_column]
        if config['min_value'] is not None:
            new_data = np.maximum(new_data, config['min_value'])
        if config['max_value'] is not None:
            new_data = np.minimum(new_data, config['max_value'])

        # Add some random noise
        noise = np.random.normal(0, parameters.get('noise', 0.1), size=len(new_data))
        new_data = new_data + noise

        # Update the column
        self.data[target_column] = new_data.astype(config['dtype'])
        self._save_state()

    def remove_column(self, name: str) -> None:
        """Remove a column from the dataset."""
        if name in self.data.columns:
            self.data = self.data.drop(columns=[name])
            self.column_configs.pop(name, None)
            # Remove any relationships involving this column
            self.column_relationships = {
                k: v for k, v in self.column_relationships.items()
                if k != name and v['source_column'] != name
            }
            self._save_state()

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column in the dataset."""
        if old_name in self.data.columns:
            self.data = self.data.rename(columns={old_name: new_name})
            if old_name in self.column_configs:
                self.column_configs[new_name] = self.column_configs.pop(old_name)
            # Update relationships
            if old_name in self.column_relationships:
                self.column_relationships[new_name] = self.column_relationships.pop(old_name)
            for rel in self.column_relationships.values():
                if rel['source_column'] == old_name:
                    rel['source_column'] = new_name
            self._save_state()

    def get_data(self) -> pd.DataFrame:
        """Get the current dataset."""
        return self.data.copy()

    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all columns."""
        return self.column_configs.copy()

    def get_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Get information about column relationships."""
        return self.column_relationships.copy()

    def adjust_length(self) -> None:
        """Adjust the length of all columns to match target_length."""
        if self.target_length is None:
            return

        current_length = len(self.data)
        if current_length < self.target_length:
            # Extend columns with NaN
            extension = pd.DataFrame(
                {col: [np.nan] * (self.target_length - current_length) 
                 for col in self.data.columns}
            )
            self.data = pd.concat([self.data, extension], ignore_index=True)
        elif current_length > self.target_length:
            # Truncate columns
            self.data = self.data.iloc[:self.target_length]

    def save_to_csv(self, filename: str) -> None:
        """Save the dataset to a CSV file."""
        self.data.to_csv(filename, index=False)

    def get_statistics(self) -> pd.DataFrame:
        """Get basic statistics for all columns."""
        return self.data.describe()

    def get_faker_providers(self) -> List[str]:
        """Get list of available Faker providers."""
        return [attr for attr in dir(self.faker) if not attr.startswith('_')]

    def get_faker_methods(self, provider: str) -> List[str]:
        """Get list of available methods for a Faker provider."""
        try:
            provider_obj = getattr(self.faker, provider)
            return [attr for attr in dir(provider_obj) if not attr.startswith('_') and callable(getattr(provider_obj, attr))]
        except AttributeError:
            return []

    def get_column_value_counts(self, column_name: str) -> pd.Series:
        """Get value counts for a categorical column."""
        if column_name in self.data.columns:
            return self.data[column_name].value_counts()
        return pd.Series()

    def _generate_datetime_data(self, num_rows: int, params: Dict[str, Any]) -> np.ndarray:
        """Generate datetime data based on parameters with time series features."""
        start_date = pd.Timestamp(params.get('start_date', pd.Timestamp.now()))
        end_date = pd.Timestamp(params.get('end_date', start_date + pd.Timedelta(days=365)))
        frequency = params.get('frequency', 'Random')
        date_format = params.get('format', 'YYYY-MM-DD')
        
        # Time series parameters
        level = params.get('level', 100.0)  # Base level
        trend = params.get('trend', 0.0)    # Trend per time unit
        seasonality = params.get('seasonality', 0.0)  # Seasonality strength
        seasonality_period = params.get('seasonality_period', 12)  # Seasonality period
        noise_level = params.get('noise_level', 0.1)  # Noise level
        cycle_period = params.get('cycle_period', 0)  # Cycle period (0 = no cycle)
        cycle_strength = params.get('cycle_strength', 0.0)  # Cycle strength
        
        # Generate base datetime series
        if frequency == 'Random':
            # Generate random dates between start and end
            date_range = (end_date - start_date).days
            random_days = np.random.randint(0, date_range + 1, size=num_rows)
            dates = [start_date + pd.Timedelta(days=int(day)) for day in random_days]
            dates.sort()  # Sort for time series analysis
        else:
            # Generate regular intervals
            if frequency == 'Daily':
                freq = 'D'
            elif frequency == 'Weekly':
                freq = 'W'
            elif frequency == 'Monthly':
                freq = 'M'
            elif frequency == 'Hourly':
                freq = 'H'
            elif frequency == 'Minutely':
                freq = 'T'
            else:
                freq = 'D'
            
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
            if len(date_range) < num_rows:
                # Repeat the pattern if needed
                dates = []
                while len(dates) < num_rows:
                    dates.extend(date_range)
                dates = dates[:num_rows]
            else:
                dates = date_range[:num_rows]
        
        # Generate time series values
        if date_format == 'Time Series with Values':
            # Check if constraints are specified
            if 'ts_min_value' in params or 'ts_max_value' in params:
                time_series_values = self._generate_time_series_values_with_constraints(
                    num_rows, params
                )
            else:
                time_series_values = self._generate_time_series_values(
                    num_rows, level, trend, seasonality, seasonality_period, 
                    noise_level, cycle_period, cycle_strength
                )
        else:
            # For non-time-series formats, we don't need values
            time_series_values = None
        
        # Combine dates with values based on format
        if date_format == 'YYYY-MM-DD':
            result = np.array([date.strftime('%Y-%m-%d') for date in dates])
        elif date_format == 'YYYY-MM-DD HH:MM:SS':
            result = np.array([date.strftime('%Y-%m-%d %H:%M:%S') for date in dates])
        elif date_format == 'Timestamp':
            result = np.array(dates)
        elif date_format == 'Unix Timestamp':
            result = np.array([int(date.timestamp()) for date in dates])
        elif date_format == 'Time Series with Values':
            # Return both dates and values
            result = pd.DataFrame({
                'datetime': dates,
                'value': time_series_values
            })
            return result
        else:
            result = np.array(dates)
        
        return result
    
    def _generate_time_series_values(self, num_rows: int, level: float, trend: float, 
                                   seasonality: float, seasonality_period: int,
                                   noise_level: float, cycle_period: int, 
                                   cycle_strength: float) -> np.ndarray:
        """Generate time series values with level, trend, seasonality, and noise."""
        time_points = np.arange(num_rows)
        
        # Base level
        values = np.full(num_rows, level)
        
        # Add trend
        if trend != 0:
            values += trend * time_points
        
        # Add seasonality
        if seasonality > 0 and seasonality_period > 1:
            seasonal_component = seasonality * np.sin(2 * np.pi * time_points / seasonality_period)
            values += seasonal_component
        
        # Add cycles (longer than seasonality)
        if cycle_strength > 0 and cycle_period > seasonality_period:
            cycle_component = cycle_strength * np.sin(2 * np.pi * time_points / cycle_period)
            values += cycle_component
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(values), num_rows)
            values += noise
        
        return values
    
    def _generate_time_series_values_with_constraints(self, num_rows: int, params: Dict[str, Any]) -> np.ndarray:
        """Generate time series values with constraints applied."""
        # Get constraint parameters
        ts_min_value = params.get('ts_min_value')
        ts_max_value = params.get('ts_max_value')
        
        # Generate base time series values
        values = self._generate_time_series_values(
            num_rows,
            params.get('level', 100.0),
            params.get('trend', 0.0),
            params.get('seasonality', 0.0),
            params.get('seasonality_period', 12),
            params.get('noise_level', 0.1),
            params.get('cycle_period', 0),
            params.get('cycle_strength', 0.0)
        )
        
        # Apply constraints
        if ts_min_value is not None:
            values = np.maximum(values, ts_min_value)
        if ts_max_value is not None:
            values = np.minimum(values, ts_max_value)
        
        return values

    def _generate_advanced_time_series_values(self, num_rows: int, params: Dict[str, Any]) -> np.ndarray:
        """Generate advanced time series with multiple seasonal patterns, events, and holiday effects."""
        time_points = np.arange(num_rows)
        
        # Base parameters
        level = params.get('level', 100.0)
        trend = params.get('trend', 0.0)
        noise_level = params.get('noise_level', 0.1)
        
        # Initialize values
        values = np.full(num_rows, level)
        
        # Add trend
        if trend != 0:
            values += trend * time_points
        
        # Add multiple seasonal patterns
        seasonal_patterns = params.get('seasonal_patterns', [])
        for pattern in seasonal_patterns:
            strength = pattern.get('strength', 0)
            period = pattern.get('period', 12)
            pattern_type = pattern.get('type', 'sine')
            
            if strength > 0 and period > 1:
                if pattern_type == 'sine':
                    seasonal_component = strength * np.sin(2 * np.pi * time_points / period)
                elif pattern_type == 'cosine':
                    seasonal_component = strength * np.cos(2 * np.pi * time_points / period)
                elif pattern_type == 'sawtooth':
                    seasonal_component = strength * (2 * (time_points / period - np.floor(time_points / period + 0.5)))
                elif pattern_type == 'square':
                    seasonal_component = strength * np.sign(np.sin(2 * np.pi * time_points / period))
                
                values += seasonal_component
        
        # Add events/spikes
        events = params.get('events', [])
        for event in events:
            position = event.get('position', 0)
            magnitude = event.get('magnitude', 0)
            duration = event.get('duration', 1)
            
            if 0 <= position < num_rows:
                start_pos = max(0, position - duration // 2)
                end_pos = min(num_rows, position + duration // 2 + 1)
                values[start_pos:end_pos] += magnitude
        
        # Add holiday effects (simplified)
        holiday_effect = params.get('holiday_effect', 0)
        if holiday_effect > 0:
            # Simulate holiday effects at regular intervals
            holiday_period = params.get('holiday_period', 30)
            holiday_positions = np.arange(0, num_rows, holiday_period)
            for pos in holiday_positions:
                if pos < num_rows:
                    values[pos] += holiday_effect
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(values), num_rows)
            values += noise
        
        return values

    def _generate_boolean_data(self, num_rows: int, params: Dict[str, Any]) -> np.ndarray:
        """Generate boolean data based on parameters."""
        p_true = params.get('p', 0.5)
        return np.random.choice([True, False], size=num_rows, p=[p_true, 1-p_true]) 