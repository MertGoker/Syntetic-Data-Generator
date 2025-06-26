import streamlit as st
import pandas as pd
import plotly.express as px
from data_generator import DataGenerator
from experimental import show_experimental_page
from data_analysis import show_data_analysis_page
from data_templates import show_data_templates_page
from column_interrelations import show_column_interrelations_page
import numpy as np

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = DataGenerator()

# Page config
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "Experimental", "Data Analysis", "Data Templates", "Column Interrelations"])

def show_enhanced_column_interface():
    """Show enhanced interface for column generation with collapsible cards."""
    st.subheader("🏷️ General Settings")
    
    # Column name
    col_name = st.text_input("Column Name", key="enhanced_col_name")
    
    # Column type selection
    column_type = st.selectbox(
        "Data Type",
        ["Numeric", "Categorical", "Boolean", "DateTime"],
        key="enhanced_col_type"
    )
    
    # Column length
    use_custom_length = st.checkbox("Use custom column length", key="enhanced_custom_length")
    if use_custom_length:
        column_length = st.number_input("Column Length", min_value=1, value=1000, key="enhanced_col_length")
    else:
        column_length = None
    
    # Initialize parameters
    params = {}
    dtype = "object"
    min_value = None
    max_value = None
    
    if column_type == "Numeric":
        st.subheader("🧪 Numeric Settings")
        
        # Distribution selection
        distribution = st.selectbox(
            "Distribution",
            ["Uniform", "Normal", "Poisson", "Triangular", "Exponential", "Binary"],
            key="enhanced_numeric_dist"
        )
        
        # Distribution parameters
        if distribution == "Binary":
            params["p"] = st.slider("Probability of 1", 0.0, 1.0, 0.5, key="enhanced_binary_p")
            dtype = "int64"
        elif distribution == "Normal":
            col1, col2 = st.columns(2)
            with col1:
                params["mean"] = st.number_input("Mean", value=0.0, key="enhanced_normal_mean")
            with col2:
                params["std"] = st.number_input("Standard Deviation", value=1.0, min_value=0.0, key="enhanced_normal_std")
        elif distribution == "Poisson":
            params["lambda"] = st.number_input("Lambda", value=1.0, min_value=0.0, key="enhanced_poisson_lambda")
        elif distribution == "Uniform":
            col1, col2 = st.columns(2)
            with col1:
                params["low"] = st.number_input("Lower Bound", value=0.0, key="enhanced_uniform_low")
            with col2:
                params["high"] = st.number_input("Upper Bound", value=1.0, key="enhanced_uniform_high")
        elif distribution == "Triangular":
            col1, col2, col3 = st.columns(3)
            with col1:
                params["left"] = st.number_input("Left Bound", value=0.0, key="enhanced_triangular_left")
            with col2:
                params["mode"] = st.number_input("Mode", value=0.5, key="enhanced_triangular_mode")
            with col3:
                params["right"] = st.number_input("Right Bound", value=1.0, key="enhanced_triangular_right")
        elif distribution == "Exponential":
            params["lambda"] = st.number_input("Lambda (rate parameter)", value=1.0, min_value=0.0, key="enhanced_exp_lambda")
        
        # Min/Max constraints
        st.write("**Value Constraints**")
        use_constraints = st.checkbox("Use min/max constraints", key="enhanced_numeric_constraints")
        if use_constraints:
            col1, col2 = st.columns(2)
            with col1:
                min_value = st.number_input("Minimum Value", value=None, key="enhanced_numeric_min")
            with col2:
                max_value = st.number_input("Maximum Value", value=None, key="enhanced_numeric_max")
        
        # Data type selection
        if distribution != "Binary":
            dtype = st.selectbox("Data Type", ["float64", "int64"], key="enhanced_numeric_dtype")
    
    elif column_type == "Categorical":
        st.subheader("🎯 Categorical Settings")
        
        # Categorical type selection
        cat_type = st.radio(
            "Type",
            ["Manual", "Faker-generated"],
            key="enhanced_cat_type"
        )
        
        if cat_type == "Manual":
            st.write("**Manual Categories**")
            values_input = st.text_area(
                "Enter Values (one per line)",
                help="Enter each possible value on a new line",
                key="enhanced_manual_values"
            )
            
            if values_input:
                values = [v.strip() for v in values_input.split('\n') if v.strip()]
                params['values'] = values
                
                # Distribution type
                distribution_type = st.radio(
                    "Distribution",
                    ["Equal", "Custom Weights"],
                    key="enhanced_manual_dist"
                )
                
                if distribution_type == "Equal":
                    params['uniform'] = True
                else:
                    params['uniform'] = False
                    st.write("**Custom Weights**")
                    st.write("Enter probability for each value (must sum to 1)")
                    
                    probs = []
                    for i, value in enumerate(values):
                        prob = st.number_input(
                            f"Probability for '{value}'",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0/len(values),
                            step=0.01,
                            format="%.2f",
                            key=f"enhanced_manual_prob_{i}"
                        )
                        probs.append(prob)
                    
                    # Validate probabilities
                    total_prob = sum(probs)
                    if not np.isclose(total_prob, 1.0, atol=0.01):
                        st.warning(f"Probabilities sum to {total_prob:.2f}, not 1.0")
                    
                    params['probabilities'] = probs
                
                distribution = "custom"
            else:
                st.warning("Please enter at least one value")
                params = {'values': [], 'uniform': True}
                distribution = "custom"
        
        else:  # Faker-generated
            st.write("**Faker-generated Categories**")
            col1, col2 = st.columns(2)
            
            with col1:
                provider = st.selectbox(
                    "Faker Provider",
                    options=st.session_state.generator.get_faker_providers(),
                    key="enhanced_faker_provider"
                )
            
            with col2:
                methods = st.session_state.generator.get_faker_methods(provider)
                faker_method = st.selectbox(
                    "Faker Method",
                    options=methods,
                    key="enhanced_faker_method"
                )
            
            params['provider'] = provider
            params['method'] = faker_method
            params['num_unique'] = st.number_input(
                "Number of Unique Values",
                min_value=1,
                max_value=1000,
                value=10,
                key="enhanced_faker_unique"
            )
            
            # Distribution type
            distribution_type = st.radio(
                "Distribution",
                ["Equal", "Custom Weights"],
                key="enhanced_faker_dist"
            )
            
            if distribution_type == "Equal":
                params['uniform'] = True
            else:
                params['uniform'] = False
                st.warning("Custom probabilities for Faker-generated values will be assigned randomly")
            
            distribution = "faker"
    
    elif column_type == "Boolean":
        st.subheader("✅ Boolean Settings")
        
        params["p"] = st.slider(
            "Probability of True",
            0.0, 1.0, 0.5,
            help="Probability of generating True values",
            key="enhanced_boolean_p"
        )
        
        distribution = "Binary"
        dtype = "bool"
    
    elif column_type == "DateTime":
        st.subheader("🗓️ DateTime Settings")
        
        # Basic datetime settings
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.now().date(),
                key="enhanced_dt_start"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=(pd.Timestamp.now() + pd.Timedelta(days=365)).date(),
                key="enhanced_dt_end"
            )
        
        # Frequency and format selection
        col1, col2 = st.columns(2)
        
        with col1:
            frequency = st.selectbox(
                "Frequency",
                ["Random", "Daily", "Weekly", "Monthly", "Hourly", "Minutely"],
                key="enhanced_dt_freq"
            )
        
        with col2:
            date_format = st.selectbox(
                "Format",
                ["YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", "Timestamp", "Unix Timestamp", "Time Series with Values"],
                key="enhanced_dt_format"
            )
        
        # Time series parameters (only show if Time Series with Values is selected)
        if date_format == "Time Series with Values":
            st.write("**📈 Time Series Parameters**")
            
            # Level and trend
            col1, col2 = st.columns(2)
            with col1:
                level = st.number_input(
                    "Base Level",
                    value=100.0,
                    help="Base value for the time series",
                    key="enhanced_dt_level"
                )
            with col2:
                trend = st.number_input(
                    "Trend per Time Unit",
                    value=0.0,
                    help="Linear trend per time unit (positive = increasing, negative = decreasing)",
                    key="enhanced_dt_trend"
                )
            
            # Seasonality
            col1, col2 = st.columns(2)
            with col1:
                seasonality = st.number_input(
                    "Seasonality Strength",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    help="Strength of seasonal pattern (0 = no seasonality)",
                    key="enhanced_dt_seasonality"
                )
            with col2:
                seasonality_period = st.number_input(
                    "Seasonality Period",
                    min_value=2,
                    max_value=365,
                    value=12,
                    help="Period of seasonality (e.g., 12 for monthly, 7 for weekly)",
                    key="enhanced_dt_seasonality_period"
                )
            
            # Cycles (longer than seasonality)
            col1, col2 = st.columns(2)
            with col1:
                cycle_strength = st.number_input(
                    "Cycle Strength",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    help="Strength of cyclical pattern (0 = no cycles)",
                    key="enhanced_dt_cycle_strength"
                )
            with col2:
                cycle_period = st.number_input(
                    "Cycle Period",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    help="Period of cycles (0 = no cycles, should be > seasonality period)",
                    key="enhanced_dt_cycle_period"
                )
            
            # Noise
            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                help="Amount of random noise to add (0 = no noise, 1 = high noise)",
                key="enhanced_dt_noise"
            )
            
            # Value constraints for time series
            st.write("**📏 Value Constraints**")
            use_ts_constraints = st.checkbox(
                "Use min/max value constraints",
                help="Constrain the generated time series values within a range",
                key="enhanced_dt_ts_constraints"
            )
            
            if use_ts_constraints:
                col1, col2 = st.columns(2)
                with col1:
                    ts_min_value = st.number_input(
                        "Minimum Value",
                        value=None,
                        help="Minimum value for the time series",
                        key="enhanced_dt_ts_min"
                    )
                with col2:
                    ts_max_value = st.number_input(
                        "Maximum Value",
                        value=None,
                        help="Maximum value for the time series",
                        key="enhanced_dt_ts_max"
                    )
                
                params['ts_min_value'] = ts_min_value
                params['ts_max_value'] = ts_max_value
                
                # Validate constraints
                if ts_min_value is not None and ts_max_value is not None:
                    if ts_min_value >= ts_max_value:
                        st.error("❌ Minimum value must be less than maximum value!")
                    else:
                        st.success(f"✅ Value range: {ts_min_value} to {ts_max_value}")
                elif ts_min_value is not None:
                    st.info(f"📏 Minimum value constraint: {ts_min_value}")
                elif ts_max_value is not None:
                    st.info(f"📏 Maximum value constraint: {ts_max_value}")
            
            # Advanced options
            st.write("**🔧 Advanced Options**")
            
            # Business hours simulation
            business_hours = st.checkbox(
                "Simulate Business Hours",
                help="Add business hours effect (higher values during weekdays 9-17)",
                key="enhanced_dt_business_hours"
            )
            
            if business_hours:
                business_effect = st.slider(
                    "Business Hours Effect",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    help="Additional value during business hours",
                    key="enhanced_dt_business_effect"
                )
                params['business_hours'] = True
                params['business_effect'] = business_effect
            
            # Weekend effects
            weekend_effect = st.checkbox(
                "Simulate Weekend Effects",
                help="Add weekend effect (different values on weekends)",
                key="enhanced_dt_weekend_effect"
            )
            
            if weekend_effect:
                weekend_magnitude = st.slider(
                    "Weekend Effect Magnitude",
                    min_value=-50.0,
                    max_value=50.0,
                    value=-5.0,
                    help="Change in value during weekends (negative = decrease, positive = increase)",
                    key="enhanced_dt_weekend_magnitude"
                )
                params['weekend_effect'] = True
                params['weekend_magnitude'] = weekend_magnitude
            
            # Multiple seasonal patterns
            multiple_seasonality = st.checkbox(
                "Add Multiple Seasonal Patterns",
                help="Add multiple seasonal patterns (e.g., daily + weekly + monthly)",
                key="enhanced_dt_multiple_seasonality"
            )
            
            if multiple_seasonality:
                st.write("**Multiple Seasonal Patterns**")
                
                # Daily pattern
                daily_strength = st.number_input(
                    "Daily Pattern Strength",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    key="enhanced_dt_daily_strength"
                )
                
                # Weekly pattern
                weekly_strength = st.number_input(
                    "Weekly Pattern Strength",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    key="enhanced_dt_weekly_strength"
                )
                
                # Monthly pattern
                monthly_strength = st.number_input(
                    "Monthly Pattern Strength",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    key="enhanced_dt_monthly_strength"
                )
                
                seasonal_patterns = []
                if daily_strength > 0:
                    seasonal_patterns.append({
                        'type': 'sine',
                        'strength': daily_strength,
                        'period': 1
                    })
                if weekly_strength > 0:
                    seasonal_patterns.append({
                        'type': 'sine',
                        'strength': weekly_strength,
                        'period': 7
                    })
                if monthly_strength > 0:
                    seasonal_patterns.append({
                        'type': 'sine',
                        'strength': monthly_strength,
                        'period': 30
                    })
                
                params['seasonal_patterns'] = seasonal_patterns
            
            # Event simulation
            add_events = st.checkbox(
                "Add Random Events/Spikes",
                help="Add random events or spikes to the time series",
                key="enhanced_dt_add_events"
            )
            
            if add_events:
                num_events = st.number_input(
                    "Number of Events",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="enhanced_dt_num_events"
                )
                
                event_magnitude = st.slider(
                    "Event Magnitude",
                    min_value=10.0,
                    max_value=100.0,
                    value=30.0,
                    help="Magnitude of events/spikes",
                    key="enhanced_dt_event_magnitude"
                )
                
                # Generate random events
                events = []
                for i in range(num_events):
                    position = np.random.randint(0, 1000)  # Will be adjusted based on actual length
                    events.append({
                        'position': position,
                        'magnitude': event_magnitude,
                        'duration': np.random.randint(1, 5)
                    })
                
                params['events'] = events
            
            # Add basic parameters to config
            params['level'] = level
            params['trend'] = trend
            params['seasonality'] = seasonality
            params['seasonality_period'] = seasonality_period
            params['cycle_strength'] = cycle_strength
            params['cycle_period'] = cycle_period
            params['noise_level'] = noise_level
        
        params['start_date'] = start_date
        params['end_date'] = end_date
        params['frequency'] = frequency
        params['format'] = date_format
        
        distribution = "datetime"
        dtype = "datetime64[ns]"
    
    return {
        'name': col_name,
        'column_type': column_type.lower(),
        'distribution': distribution,
        'parameters': params,
        'dtype': dtype,
        'length': column_length,
        'min_value': min_value,
        'max_value': max_value
    }

if page == "Main":
    st.title("Synthetic Data Generator")

    # Sidebar for dataset length
    with st.sidebar:
        st.header("Dataset Configuration")
        target_length = st.number_input("Target Dataset Length", min_value=1, value=1000)
        if st.button("Set Length"):
            st.session_state.generator.set_target_length(target_length)
            st.session_state.generator.adjust_length()
            st.success(f"Dataset length set to {target_length}")

    # Add new column section
    st.header("➕ Add New Column")
    
    # Use enhanced interface
    column_config = show_enhanced_column_interface()
    
    # Add column button
    if st.button("Add Column (+)", type="primary"):
        if column_config['name']:
            try:
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
                st.success(f"Column '{column_config['name']}' added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding column: {str(e)}")
        else:
            st.warning("Please enter a column name")

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        # Column management with collapsible cards
        st.header("📋 Column Management")
        current_columns = list(st.session_state.generator.data.columns)
        if current_columns:
            for col in current_columns:
                config = st.session_state.generator.column_configs[col]
                
                # Create collapsible card for each column
                with st.expander(f"📊 {col} ({config['column_type'].title()})", expanded=False):
                    st.write("**🏷️ General Settings**")
                    st.write(f"**Column Name:** {col}")
                    st.write(f"**Data Type:** {config['dtype']}")
                    
                    # Display settings based on column type
                    if config['column_type'] == 'numeric':
                        st.write("**🧪 Numeric Settings**")
                        st.write(f"**Distribution:** {config['distribution']}")
                        
                        if config['distribution'] == 'Normal':
                            st.write(f"**Mean:** {config['parameters'].get('mean', 'N/A')}")
                            st.write(f"**Std Dev:** {config['parameters'].get('std', 'N/A')}")
                        elif config['distribution'] == 'Uniform':
                            st.write(f"**Range:** {config['parameters'].get('low', 'N/A')} to {config['parameters'].get('high', 'N/A')}")
                        elif config['distribution'] == 'Poisson':
                            st.write(f"**Lambda:** {config['parameters'].get('lambda', 'N/A')}")
                        elif config['distribution'] == 'Triangular':
                            st.write(f"**Bounds:** {config['parameters'].get('left', 'N/A')}, {config['parameters'].get('mode', 'N/A')}, {config['parameters'].get('right', 'N/A')}")
                        elif config['distribution'] == 'Exponential':
                            st.write(f"**Lambda:** {config['parameters'].get('lambda', 'N/A')}")
                        elif config['distribution'] == 'Binary':
                            st.write(f"**Probability of 1:** {config['parameters'].get('p', 'N/A')}")
                        
                        if config['min_value'] is not None or config['max_value'] is not None:
                            st.write(f"**Constraints:** Min={config['min_value']}, Max={config['max_value']}")
                    
                    elif config['column_type'] == 'categorical':
                        st.write("**🎯 Categorical Settings**")
                        if config['distribution'] == 'faker':
                            st.write(f"**Type:** Faker-generated")
                            st.write(f"**Provider:** {config['parameters'].get('provider', 'N/A')}")
                            st.write(f"**Method:** {config['parameters'].get('method', 'N/A')}")
                            st.write(f"**Unique Values:** {config['parameters'].get('num_unique', 'N/A')}")
                        else:
                            st.write(f"**Type:** Manual")
                            st.write(f"**Values:** {', '.join(config['parameters'].get('values', [])[:5])}{'...' if len(config['parameters'].get('values', [])) > 5 else ''}")
                        
                        st.write(f"**Distribution:** {'Equal' if config['parameters'].get('uniform', True) else 'Custom Weights'}")
                    
                    elif config['column_type'] == 'boolean':
                        st.write("**✅ Boolean Settings**")
                        st.write(f"**Probability of True:** {config['parameters'].get('p', 'N/A')}")
                    
                    elif config['column_type'] == 'datetime':
                        st.write("**🗓️ DateTime Settings**")
                        st.write(f"**Range:** {config['parameters'].get('start_date', 'N/A')} to {config['parameters'].get('end_date', 'N/A')}")
                        st.write(f"**Frequency:** {config['parameters'].get('frequency', 'N/A')}")
                        st.write(f"**Format:** {config['parameters'].get('format', 'N/A')}")
                        
                        # Show time series parameters if they exist
                        if config['parameters'].get('level') is not None:
                            st.write("**📈 Time Series Parameters**")
                            st.write(f"**Base Level:** {config['parameters'].get('level', 'N/A')}")
                            st.write(f"**Trend:** {config['parameters'].get('trend', 'N/A')}")
                            st.write(f"**Seasonality Strength:** {config['parameters'].get('seasonality', 'N/A')}")
                            st.write(f"**Seasonality Period:** {config['parameters'].get('seasonality_period', 'N/A')}")
                            st.write(f"**Cycle Strength:** {config['parameters'].get('cycle_strength', 'N/A')}")
                            st.write(f"**Cycle Period:** {config['parameters'].get('cycle_period', 'N/A')}")
                            st.write(f"**Noise Level:** {config['parameters'].get('noise_level', 'N/A')}")
                            
                            # Show value constraints if they exist
                            if config['parameters'].get('ts_min_value') is not None or config['parameters'].get('ts_max_value') is not None:
                                st.write("**📏 Value Constraints**")
                                st.write(f"**Min Value:** {config['parameters'].get('ts_min_value', 'N/A')}")
                                st.write(f"**Max Value:** {config['parameters'].get('ts_max_value', 'N/A')}")
                    
                    # Column actions
                    st.write("**🔧 Actions**")
                    new_name = st.text_input(f"New name for {col}", value=col, key=f"rename_{col}")
                    
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if new_name != col and st.button(f"Rename", key=f"rename_btn_{col}"):
                            st.session_state.generator.rename_column(col, new_name)
                            st.success(f"Column renamed to {new_name}")
                            st.rerun()
                    
                    with action_col2:
                        if st.button(f"Delete", key=f"delete_btn_{col}", type="secondary"):
                            st.session_state.generator.remove_column(col)
                            st.success(f"Column {col} deleted")
                            st.rerun()

    with col2:
        st.header("📊 Data Preview")
        
        # Display the current dataset
        if not st.session_state.generator.data.empty:
            st.dataframe(st.session_state.generator.data)
            
            # Display statistics
            st.subheader("📈 Data Statistics")
            if any(st.session_state.generator.column_configs[col]['column_type'] == 'categorical' 
                  for col in st.session_state.generator.data.columns):
                # For datasets with categorical columns, show value counts
                for col in st.session_state.generator.data.columns:
                    if st.session_state.generator.column_configs[col]['column_type'] == 'categorical':
                        st.write(f"Value counts for {col}:")
                        st.dataframe(st.session_state.generator.get_column_value_counts(col))
            else:
                # For numeric-only datasets, show standard statistics
                st.dataframe(st.session_state.generator.get_statistics())
            
            # Plot distributions
            st.subheader("📊 Distribution Plots")
            for col in st.session_state.generator.data.columns:
                config = st.session_state.generator.column_configs[col]
                if config['column_type'] == 'categorical':
                    # For categorical columns, show a bar plot of value counts
                    value_counts = st.session_state.generator.get_column_value_counts(col)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                elif config['column_type'] == 'boolean':
                    # For boolean columns, show a bar plot
                    true_count = (st.session_state.generator.data[col] == True).sum()
                    false_count = (st.session_state.generator.data[col] == False).sum()
                    fig = px.bar(
                        pd.DataFrame({
                            'Value': ['True', 'False'],
                            'Count': [true_count, false_count]
                        }),
                        x='Value',
                        y='Count',
                        title=f"Distribution of {col}"
                    )
                elif config['distribution'] == 'binary':
                    # For binary columns, show a bar plot
                    fig = px.bar(
                        pd.DataFrame({
                            'Value': ['0', '1'],
                            'Count': [
                                (st.session_state.generator.data[col] == 0).sum(),
                                (st.session_state.generator.data[col] == 1).sum()
                            ]
                        }),
                        x='Value',
                        y='Count',
                        title=f"Distribution of {col}"
                    )
                elif config['column_type'] == 'datetime':
                    # For datetime columns, show a histogram
                    fig = px.histogram(
                        st.session_state.generator.data,
                        x=col,
                        title=f"Distribution of {col}",
                        nbins=20
                    )
                    
                    # Add temporal distribution plot
                    st.write(f"**📅 Temporal Distribution for {col}**")
                    
                    # Convert to datetime if it's not already
                    datetime_data = pd.to_datetime(st.session_state.generator.data[col])
                    
                    # Create temporal distribution plot
                    fig_temp = px.scatter(
                        x=datetime_data,
                        y=range(len(datetime_data)),
                        title=f"Temporal Distribution: {col}",
                        labels={'x': col, 'y': 'Index'}
                    )
                    fig_temp.update_layout(
                        xaxis_title=col,
                        yaxis_title="Data Point Index",
                        showlegend=False
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                    
                    # Show datetime statistics
                    st.write(f"**📊 DateTime Statistics for {col}**")
                    datetime_stats = {
                        'Start Date': datetime_data.min(),
                        'End Date': datetime_data.max(),
                        'Duration': datetime_data.max() - datetime_data.min(),
                        'Total Points': len(datetime_data),
                        'Unique Dates': datetime_data.nunique(),
                        'Most Common Date': datetime_data.mode().iloc[0] if len(datetime_data.mode()) > 0 else 'N/A'
                    }
                    st.dataframe(pd.DataFrame(list(datetime_stats.items()), columns=['Metric', 'Value']))
                    
                    # Check if there's an associated value column for time series plotting
                    value_col = f"{col}_value"
                    if value_col in st.session_state.generator.data.columns:
                        st.write(f"**📈 Time Series Plot for {col}**")
                        
                        # Create time series plot
                        time_series_data = st.session_state.generator.data[[col, value_col]].copy()
                        time_series_data = time_series_data.sort_values(col)  # Sort by datetime
                        
                        fig_ts = px.line(
                            time_series_data,
                            x=col,
                            y=value_col,
                            title=f"Time Series: {value_col} over {col}",
                            labels={'x': col, 'y': value_col}
                        )
                        
                        # Add trend line if there's a trend
                        config_ts = st.session_state.generator.column_configs[value_col]
                        if config_ts.get('parameters', {}).get('trend', 0) != 0:
                            # Calculate trend line
                            x_numeric = np.arange(len(time_series_data))
                            trend_line = (config_ts['parameters']['trend'] * x_numeric + 
                                        config_ts['parameters'].get('level', 100))
                            
                            fig_ts.add_scatter(
                                x=time_series_data[col],
                                y=trend_line,
                                mode='lines',
                                name='Trend Line',
                                line=dict(dash='dash', color='red')
                            )
                        
                        fig_ts.update_layout(
                            xaxis_title=col,
                            yaxis_title=value_col,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Show time series statistics
                        st.write(f"**📊 Time Series Statistics for {value_col}**")
                        ts_stats = time_series_data[value_col].describe()
                        st.dataframe(ts_stats)
                        
                        # Advanced time series analysis
                        st.write(f"**🔍 Advanced Time Series Analysis**")
                        
                        # Autocorrelation plot
                        try:
                            from statsmodels.graphics.tsaplots import plot_acf
                            import matplotlib.pyplot as plt
                            
                            # Create autocorrelation plot
                            fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                            plot_acf(time_series_data[value_col].dropna(), ax=ax_acf, lags=40)
                            ax_acf.set_title(f'Autocorrelation Function for {value_col}')
                            ax_acf.set_xlabel('Lag')
                            ax_acf.set_ylabel('Autocorrelation')
                            st.pyplot(fig_acf)
                            plt.close()
                            
                        except ImportError:
                            st.info("Install statsmodels for autocorrelation analysis: `pip install statsmodels`")
                        
                        # Seasonality analysis
                        if config_ts.get('parameters', {}).get('seasonality', 0) > 0:
                            st.write(f"**📅 Seasonality Analysis**")
                            seasonality_period = config_ts['parameters'].get('seasonality_period', 12)
                            
                            # Create seasonal subplot
                            if len(time_series_data) >= seasonality_period * 2:
                                # Reshape data for seasonal analysis
                                seasonal_data = time_series_data[value_col].values
                                n_seasons = len(seasonal_data) // seasonality_period
                                
                                if n_seasons > 0:
                                    # Create seasonal plot
                                    seasonal_reshaped = seasonal_data[:n_seasons * seasonality_period].reshape(n_seasons, seasonality_period)
                                    seasonal_means = np.mean(seasonal_reshaped, axis=0)
                                    
                                    fig_seasonal = px.line(
                                        x=range(1, seasonality_period + 1),
                                        y=seasonal_means,
                                        title=f"Seasonal Pattern (Period: {seasonality_period})",
                                        labels={'x': 'Period Position', 'y': 'Average Value'}
                                    )
                                    fig_seasonal.update_layout(
                                        xaxis_title="Position in Season",
                                        yaxis_title="Average Value"
                                    )
                                    st.plotly_chart(fig_seasonal, use_container_width=True)
                        
                        # Trend analysis
                        if config_ts.get('parameters', {}).get('trend', 0) != 0:
                            st.write(f"**📈 Trend Analysis**")
                            trend_value = config_ts['parameters']['trend']
                            trend_direction = "Increasing" if trend_value > 0 else "Decreasing"
                            st.write(f"**Trend Direction:** {trend_direction}")
                            st.write(f"**Trend Magnitude:** {trend_value:.4f} per time unit")
                            
                            # Calculate trend strength
                            total_change = trend_value * len(time_series_data)
                            initial_value = config_ts['parameters'].get('level', 100)
                            trend_strength = abs(total_change / initial_value) * 100
                            st.write(f"**Trend Strength:** {trend_strength:.2f}% change over the period")
                        
                        # Noise analysis
                        noise_level = config_ts.get('parameters', {}).get('noise_level', 0)
                        if noise_level > 0:
                            st.write(f"**🔊 Noise Analysis**")
                            st.write(f"**Noise Level:** {noise_level:.3f}")
                            
                            # Calculate signal-to-noise ratio
                            signal_std = time_series_data[value_col].std()
                            noise_std = signal_std * noise_level
                            snr = signal_std / noise_std if noise_std > 0 else float('inf')
                            st.write(f"**Signal-to-Noise Ratio:** {snr:.2f}")
                else:
                    # For other numeric columns, show a histogram
                    fig = px.histogram(
                        st.session_state.generator.data,
                        x=col,
                        title=f"Distribution of {col}"
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            # Save to CSV
            st.subheader("💾 Save Dataset")
            filename = st.text_input(
                "Filename",
                value="synthetic_data.csv",
                help="Enter the name for your CSV file (e.g., my_data.csv)"
            )
            
            if st.button("Save to CSV", use_container_width=True):
                if not filename.endswith('.csv'):
                    filename += '.csv'
                try:
                    st.session_state.generator.save_to_csv(filename)
                    st.success(f"Data saved to {filename}")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        else:
            st.info("No data generated yet. Add columns using the controls on the left.")

elif page == "Experimental":
    show_experimental_page()

elif page == "Data Analysis":
    show_data_analysis_page()

elif page == "Data Templates":
    show_data_templates_page()

elif page == "Column Interrelations":
    show_column_interrelations_page() 