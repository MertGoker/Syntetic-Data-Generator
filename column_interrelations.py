import streamlit as st
import pandas as pd
import numpy as np
from data_generator import DataGenerator
from typing import Dict, List, Any, Tuple
import re

class ColumnInterrelations:
    def __init__(self, generator: DataGenerator):
        self.generator = generator
        self.interrelations = {}
    
    def get_column_types(self) -> Dict[str, str]:
        """Get column types for all columns."""
        column_types = {}
        for col in self.generator.data.columns:
            if col in self.generator.column_configs:
                column_types[col] = self.generator.column_configs[col]['column_type']
            else:
                # Infer type from data
                if self.generator.data[col].dtype in ['object', 'string']:
                    column_types[col] = 'categorical'
                elif self.generator.data[col].dtype in ['bool']:
                    column_types[col] = 'boolean'
                elif pd.api.types.is_datetime64_any_dtype(self.generator.data[col]):
                    column_types[col] = 'datetime'
                else:
                    column_types[col] = 'numeric'
        return column_types
    
    def get_relationship_types(self, source_type: str, target_type: str) -> List[str]:
        """Get available relationship types for given column type combination."""
        if source_type == 'numeric' and target_type == 'numeric':
            return [
                'Mathematical Operations',
                'Conditional Mapping',
                'Statistical Functions',
                'Custom Formula'
            ]
        elif source_type == 'categorical' and target_type == 'categorical':
            return [
                'Category Mapping',
                'Conditional Categories',
                'Combined Categories',
                'Frequency-based'
            ]
        elif source_type == 'numeric' and target_type == 'categorical':
            return [
                'Binning',
                'Threshold-based Categories',
                'Percentile-based Categories',
                'Custom Rules'
            ]
        elif source_type == 'categorical' and target_type == 'numeric':
            return [
                'Category Averages',
                'Category Counts',
                'Encoded Values',
                'Custom Assignments'
            ]
        elif source_type == 'boolean' or target_type == 'boolean':
            return [
                'Logical Operations',
                'Conditional Boolean',
                'Threshold-based Boolean'
            ]
        else:
            return ['Custom Relationship']
    
    def apply_numeric_numeric_relationship(self, source_col: str, target_col: str, 
                                         relationship_type: str, params: Dict) -> None:
        """Apply numeric to numeric relationship."""
        source_data = self.generator.data[source_col].values
        
        if relationship_type == 'Mathematical Operations':
            operation = params.get('operation', 'add')
            value = params.get('value', 0)
            
            if operation == 'add':
                new_data = source_data + value
            elif operation == 'subtract':
                new_data = source_data - value
            elif operation == 'multiply':
                new_data = source_data * value
            elif operation == 'divide':
                new_data = source_data / (value if value != 0 else 1)
            elif operation == 'power':
                new_data = source_data ** value
            elif operation == 'sqrt':
                new_data = np.sqrt(np.abs(source_data))
            elif operation == 'log':
                new_data = np.log(np.abs(source_data) + 1e-8)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        elif relationship_type == 'Conditional Mapping':
            condition = params.get('condition', 'greater_than')
            threshold = params.get('threshold', 0)
            true_value = params.get('true_value', 1)
            false_value = params.get('false_value', 0)
            
            if condition == 'greater_than':
                new_data = np.where(source_data > threshold, true_value, false_value)
            elif condition == 'less_than':
                new_data = np.where(source_data < threshold, true_value, false_value)
            elif condition == 'equal_to':
                new_data = np.where(source_data == threshold, true_value, false_value)
            elif condition == 'greater_equal':
                new_data = np.where(source_data >= threshold, true_value, false_value)
            elif condition == 'less_equal':
                new_data = np.where(source_data <= threshold, true_value, false_value)
            else:
                raise ValueError(f"Unsupported condition: {condition}")
        
        elif relationship_type == 'Statistical Functions':
            function = params.get('function', 'z_score')
            
            if function == 'z_score':
                mean_val = np.mean(source_data)
                std_val = np.std(source_data)
                new_data = (source_data - mean_val) / (std_val if std_val != 0 else 1)
            elif function == 'min_max_scale':
                min_val = np.min(source_data)
                max_val = np.max(source_data)
                new_data = (source_data - min_val) / (max_val - min_val if max_val != min_val else 1)
            elif function == 'robust_scale':
                median_val = np.median(source_data)
                q75, q25 = np.percentile(source_data, [75, 25])
                iqr = q75 - q25
                new_data = (source_data - median_val) / (iqr if iqr != 0 else 1)
            else:
                raise ValueError(f"Unsupported function: {function}")
        
        elif relationship_type == 'Custom Formula':
            formula = params.get('formula', 'x')
            # Replace 'x' with source_data in the formula
            if formula == 'x':
                new_data = source_data
            elif formula == 'x**2':
                new_data = source_data ** 2
            elif formula == 'sqrt(x)':
                new_data = np.sqrt(np.abs(source_data))
            elif formula == 'log(x)':
                new_data = np.log(np.abs(source_data) + 1e-8)
            else:
                # Try to evaluate the formula (basic implementation)
                try:
                    x = source_data
                    new_data = eval(formula)
                except:
                    raise ValueError(f"Invalid formula: {formula}")
        
        # Apply constraints
        config = self.generator.column_configs[target_col]
        if config['min_value'] is not None:
            new_data = np.maximum(new_data, config['min_value'])
        if config['max_value'] is not None:
            new_data = np.minimum(new_data, config['max_value'])
        
        # Add noise if specified
        noise_level = params.get('noise_level', 0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(new_data), len(new_data))
            new_data = new_data + noise
        
        # Update the column
        self.generator.data[target_col] = new_data.astype(config['dtype'])
    
    def apply_categorical_categorical_relationship(self, source_col: str, target_col: str,
                                                 relationship_type: str, params: Dict) -> None:
        """Apply categorical to categorical relationship."""
        source_data = self.generator.data[source_col].values
        
        if relationship_type == 'Category Mapping':
            mapping = params.get('mapping', {})
            default_value = params.get('default_value', 'Unknown')
            
            new_data = []
            for value in source_data:
                new_data.append(mapping.get(str(value), default_value))
            new_data = np.array(new_data)
        
        elif relationship_type == 'Conditional Categories':
            condition_col = params.get('condition_column', source_col)
            condition_value = params.get('condition_value', '')
            true_category = params.get('true_category', 'Yes')
            false_category = params.get('false_category', 'No')
            
            condition_data = self.generator.data[condition_col].values
            new_data = np.where(condition_data == condition_value, true_category, false_category)
        
        elif relationship_type == 'Combined Categories':
            separator = params.get('separator', '_')
            new_data = []
            for value in source_data:
                new_data.append(f"{value}{separator}Combined")
            new_data = np.array(new_data)
        
        elif relationship_type == 'Frequency-based':
            value_counts = self.generator.data[source_col].value_counts()
            threshold = params.get('threshold', 10)
            
            new_data = []
            for value in source_data:
                count = value_counts.get(value, 0)
                if count > threshold:
                    new_data.append(f"{value}_Frequent")
                else:
                    new_data.append(f"{value}_Rare")
            new_data = np.array(new_data)
        
        # Update the column
        self.generator.data[target_col] = new_data
    
    def apply_numeric_categorical_relationship(self, source_col: str, target_col: str,
                                             relationship_type: str, params: Dict) -> None:
        """Apply numeric to categorical relationship."""
        source_data = self.generator.data[source_col].values
        
        if relationship_type == 'Binning':
            bins = params.get('bins', 5)
            labels = params.get('labels', [])
            
            if len(labels) == 0:
                labels = [f'Bin_{i+1}' for i in range(bins)]
            
            new_data = pd.cut(source_data, bins=bins, labels=labels, include_lowest=True)
        
        elif relationship_type == 'Threshold-based Categories':
            thresholds = params.get('thresholds', [])
            categories = params.get('categories', [])
            
            if len(thresholds) + 1 != len(categories):
                raise ValueError("Number of categories must be number of thresholds + 1")
            
            new_data = []
            for value in source_data:
                category_index = 0
                for i, threshold in enumerate(thresholds):
                    if value > threshold:
                        category_index = i + 1
                new_data.append(categories[category_index])
            new_data = np.array(new_data)
        
        elif relationship_type == 'Percentile-based Categories':
            percentiles = params.get('percentiles', [25, 50, 75])
            categories = params.get('categories', ['Low', 'Medium', 'High', 'Very High'])
            
            percentile_values = np.percentile(source_data, percentiles)
            
            new_data = []
            for value in source_data:
                category_index = 0
                for i, p_val in enumerate(percentile_values):
                    if value > p_val:
                        category_index = i + 1
                new_data.append(categories[category_index])
            new_data = np.array(new_data)
        
        elif relationship_type == 'Custom Rules':
            rules = params.get('rules', [])
            default_category = params.get('default_category', 'Other')
            
            new_data = []
            for value in source_data:
                category = default_category
                for rule in rules:
                    condition = rule.get('condition', '')
                    rule_category = rule.get('category', default_category)
                    
                    if condition == 'greater_than' and value > rule.get('value', 0):
                        category = rule_category
                    elif condition == 'less_than' and value < rule.get('value', 0):
                        category = rule_category
                    elif condition == 'between' and rule.get('min', 0) <= value <= rule.get('max', 0):
                        category = rule_category
                new_data.append(category)
            new_data = np.array(new_data)
        
        # Update the column
        self.generator.data[target_col] = new_data
    
    def apply_categorical_numeric_relationship(self, source_col: str, target_col: str,
                                             relationship_type: str, params: Dict) -> None:
        """Apply categorical to numeric relationship."""
        source_data = self.generator.data[source_col].values
        
        if relationship_type == 'Category Averages':
            # Calculate average of another numeric column for each category
            numeric_col = params.get('numeric_column', '')
            if numeric_col not in self.generator.data.columns:
                raise ValueError(f"Numeric column {numeric_col} not found")
            
            numeric_data = self.generator.data[numeric_col].values
            category_means = {}
            
            for category in np.unique(source_data):
                mask = source_data == category
                category_means[category] = np.mean(numeric_data[mask])
            
            new_data = np.array([category_means.get(cat, 0) for cat in source_data])
        
        elif relationship_type == 'Category Counts':
            # Count occurrences of each category
            value_counts = self.generator.data[source_col].value_counts()
            new_data = np.array([value_counts.get(cat, 0) for cat in source_data])
        
        elif relationship_type == 'Encoded Values':
            # Simple label encoding
            unique_categories = np.unique(source_data)
            category_map = {cat: i for i, cat in enumerate(unique_categories)}
            new_data = np.array([category_map.get(cat, 0) for cat in source_data])
        
        elif relationship_type == 'Custom Assignments':
            assignments = params.get('assignments', {})
            default_value = params.get('default_value', 0)
            
            new_data = np.array([assignments.get(str(cat), default_value) for cat in source_data])
        
        # Apply constraints
        config = self.generator.column_configs[target_col]
        if config['min_value'] is not None:
            new_data = np.maximum(new_data, config['min_value'])
        if config['max_value'] is not None:
            new_data = np.minimum(new_data, config['max_value'])
        
        # Update the column
        self.generator.data[target_col] = new_data.astype(config['dtype'])
    
    def apply_calculated_relationship(self, target_col: str, formula: str, source_cols: List[str]) -> None:
        """Apply a calculated relationship using a formula with multiple source columns."""
        try:
            # Create a copy of the data for calculation
            calc_data = self.generator.data[source_cols].copy()
            
            # Replace column names in formula with actual data
            for col in source_cols:
                formula = formula.replace(col, f"calc_data['{col}']")
            
            # Evaluate the formula
            new_data = eval(formula)
            
            # Apply constraints
            config = self.generator.column_configs[target_col]
            if config['min_value'] is not None:
                new_data = np.maximum(new_data, config['min_value'])
            if config['max_value'] is not None:
                new_data = np.minimum(new_data, config['max_value'])
            
            # Update the column
            self.generator.data[target_col] = new_data.astype(config['dtype'])
            
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {str(e)}")

def show_column_interrelations_page():
    """Show the column interrelations page."""
    st.title("🔗 Column Interrelations")
    st.write("Create relationships between columns with conditional statements and calculated formulas.")
    
    if 'generator' not in st.session_state or st.session_state.generator.data.empty:
        st.error("Please generate some data in the main page first.")
        return
    
    # Initialize interrelations
    if 'interrelations' not in st.session_state:
        st.session_state.interrelations = ColumnInterrelations(st.session_state.generator)
    
    interrelations = st.session_state.interrelations
    column_types = interrelations.get_column_types()
    
    # Page tabs
    tab1, tab2, tab3 = st.tabs(["🔗 Simple Relationships", "🧮 Calculated Formulas", "📊 Relationship Overview"])
    
    with tab1:
        st.subheader("Create Column Relationships")
        
        # Column selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_column = st.selectbox(
                "Source Column",
                options=list(st.session_state.generator.data.columns),
                key="interrel_source"
            )
        
        with col2:
            target_column = st.selectbox(
                "Target Column",
                options=list(st.session_state.generator.data.columns),
                index=1 if len(st.session_state.generator.data.columns) > 1 else 0,
                key="interrel_target"
            )
        
        if source_column and target_column and source_column != target_column:
            source_type = column_types[source_column]
            target_type = column_types[target_column]
            
            st.write(f"**Relationship:** {source_type.title()} → {target_type.title()}")
            
            # Get available relationship types
            relationship_types = interrelations.get_relationship_types(source_type, target_type)
            
            relationship_type = st.selectbox(
                "Relationship Type",
                options=relationship_types,
                key="interrel_type"
            )
            
            # Show relationship-specific parameters
            params = show_relationship_parameters(relationship_type, source_type, target_type, 
                                                source_column, target_column)
            
            if st.button("Apply Relationship", type="primary"):
                try:
                    apply_relationship(interrelations, source_column, target_column, 
                                     relationship_type, params, source_type, target_type)
                    st.success(f"Relationship applied successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error applying relationship: {str(e)}")
    
    with tab2:
        st.subheader("Calculated Formulas")
        st.write("Create calculated columns using formulas with multiple source columns.")
        
        # Formula input
        formula = st.text_area(
            "Formula",
            placeholder="Example: height / (weight ** 0.5) for BMI calculation",
            help="Use column names directly in the formula. Supported operations: +, -, *, /, **, sqrt(), log(), etc."
        )
        
        # Target column name
        target_col_name = st.text_input(
            "Target Column Name",
            placeholder="e.g., BMI, Total_Score, etc."
        )
        
        # Source columns selection
        source_cols = st.multiselect(
            "Source Columns",
            options=list(st.session_state.generator.data.columns),
            help="Select columns to use in the formula"
        )
        
        if st.button("Create Calculated Column", type="primary"):
            if formula and target_col_name and source_cols:
                try:
                    # Create new column if it doesn't exist
                    if target_col_name not in st.session_state.generator.data.columns:
                        st.session_state.generator.add_column(
                            name=target_col_name,
                            distribution="calculated",
                            parameters={},
                            dtype="float64",
                            column_type="numeric"
                        )
                    
                    # Apply calculated relationship
                    interrelations.apply_calculated_relationship(target_col_name, formula, source_cols)
                    st.success(f"Calculated column '{target_col_name}' created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating calculated column: {str(e)}")
            else:
                st.warning("Please provide formula, target column name, and select source columns.")
    
    with tab3:
        st.subheader("Current Relationships")
        
        # Show existing relationships
        relationships = st.session_state.generator.get_relationships()
        if relationships:
            for target, rel in relationships.items():
                with st.expander(f"{target} ← {rel['source_column']} ({rel['relationship_type']})"):
                    st.write("**Parameters:**", rel['parameters'])
                    if st.button(f"Remove Relationship for {target}", key=f"remove_rel_{target}"):
                        # Remove relationship by regenerating the column
                        config = st.session_state.generator.column_configs[target]
                        st.session_state.generator.add_column(
                            name=target,
                            distribution=config['distribution'],
                            parameters=config['parameters'],
                            dtype=config['dtype'],
                            length=config['length'],
                            min_value=config['min_value'],
                            max_value=config['max_value'],
                            column_type=config['column_type']
                        )
                        st.success(f"Relationship removed for {target}")
                        st.rerun()
        else:
            st.info("No relationships defined yet.")

def show_relationship_parameters(relationship_type: str, source_type: str, target_type: str,
                               source_column: str, target_column: str) -> Dict:
    """Show parameters for the selected relationship type."""
    params = {}
    
    st.write(f"**Parameters for {relationship_type}**")
    
    if relationship_type == 'Mathematical Operations':
        col1, col2 = st.columns(2)
        with col1:
            params['operation'] = st.selectbox(
                "Operation",
                ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'log'],
                key="math_op"
            )
        with col2:
            if params['operation'] not in ['sqrt', 'log']:
                params['value'] = st.number_input("Value", value=1.0, key="math_val")
    
    elif relationship_type == 'Conditional Mapping':
        col1, col2 = st.columns(2)
        with col1:
            params['condition'] = st.selectbox(
                "Condition",
                ['greater_than', 'less_than', 'equal_to', 'greater_equal', 'less_equal'],
                key="cond_type"
            )
        with col2:
            params['threshold'] = st.number_input("Threshold", value=0.0, key="cond_thresh")
        
        col1, col2 = st.columns(2)
        with col1:
            params['true_value'] = st.number_input("True Value", value=1.0, key="cond_true")
        with col2:
            params['false_value'] = st.number_input("False Value", value=0.0, key="cond_false")
    
    elif relationship_type == 'Statistical Functions':
        params['function'] = st.selectbox(
            "Function",
            ['z_score', 'min_max_scale', 'robust_scale'],
            key="stat_func"
        )
    
    elif relationship_type == 'Custom Formula':
        params['formula'] = st.text_input(
            "Formula",
            value="x",
            help="Use 'x' to represent the source column. Examples: x**2, sqrt(x), log(x)"
        )
    
    elif relationship_type == 'Category Mapping':
        st.write("**Category Mapping**")
        st.write("Enter the mapping from source values to target values:")
        
        # Get unique values from source column
        source_values = st.session_state.generator.data[source_column].unique()
        mapping = {}
        
        for value in source_values[:10]:  # Limit to first 10 values
            target_value = st.text_input(
                f"Map '{value}' to:",
                value=str(value),
                key=f"map_{value}"
            )
            mapping[str(value)] = target_value
        
        params['mapping'] = mapping
        params['default_value'] = st.text_input("Default value for unmapped values:", value="Unknown")
    
    elif relationship_type == 'Conditional Categories':
        col1, col2 = st.columns(2)
        with col1:
            params['condition_value'] = st.text_input("Condition Value", key="cond_cat_val")
        with col2:
            params['true_category'] = st.text_input("True Category", value="Yes", key="cond_cat_true")
        
        params['false_category'] = st.text_input("False Category", value="No", key="cond_cat_false")
    
    elif relationship_type == 'Combined Categories':
        params['separator'] = st.text_input("Separator", value="_", key="combine_sep")
    
    elif relationship_type == 'Frequency-based':
        params['threshold'] = st.number_input("Frequency Threshold", value=10, min_value=1, key="freq_thresh")
    
    elif relationship_type == 'Binning':
        col1, col2 = st.columns(2)
        with col1:
            params['bins'] = st.number_input("Number of Bins", min_value=2, value=5, key="bin_count")
        with col2:
            bin_labels = st.text_input(
                "Bin Labels (comma-separated)",
                help="Leave empty for auto-generated labels",
                key="bin_labels"
            )
            if bin_labels:
                params['labels'] = [label.strip() for label in bin_labels.split(',')]
    
    elif relationship_type == 'Threshold-based Categories':
        thresholds_input = st.text_input(
            "Thresholds (comma-separated)",
            help="e.g., 25, 50, 75",
            key="thresh_values"
        )
        if thresholds_input:
            params['thresholds'] = [float(t.strip()) for t in thresholds_input.split(',')]
        
        categories_input = st.text_input(
            "Categories (comma-separated)",
            help="e.g., Low, Medium, High, Very High",
            key="thresh_cats"
        )
        if categories_input:
            params['categories'] = [cat.strip() for cat in categories_input.split(',')]
    
    elif relationship_type == 'Percentile-based Categories':
        col1, col2 = st.columns(2)
        with col1:
            percentiles_input = st.text_input(
                "Percentiles (comma-separated)",
                value="25, 50, 75",
                help="e.g., 25, 50, 75",
                key="percentile_vals"
            )
            if percentiles_input:
                params['percentiles'] = [float(p.strip()) for p in percentiles_input.split(',')]
        
        with col2:
            categories_input = st.text_input(
                "Categories (comma-separated)",
                value="Low, Medium, High, Very High",
                help="e.g., Low, Medium, High, Very High",
                key="percentile_cats"
            )
            if categories_input:
                params['categories'] = [cat.strip() for cat in categories_input.split(',')]
    
    elif relationship_type == 'Custom Rules':
        st.write("**Custom Rules**")
        st.write("Define rules for categorization:")
        
        rules = []
        num_rules = st.number_input("Number of rules", min_value=1, max_value=5, value=2, key="num_rules")
        
        for i in range(num_rules):
            st.write(f"**Rule {i+1}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                condition = st.selectbox(
                    "Condition",
                    ['greater_than', 'less_than', 'between'],
                    key=f"rule_cond_{i}"
                )
            
            with col2:
                if condition == 'between':
                    min_val = st.number_input("Min Value", key=f"rule_min_{i}")
                    max_val = st.number_input("Max Value", key=f"rule_max_{i}")
                    rule = {'condition': condition, 'min': min_val, 'max': max_val}
                else:
                    value = st.number_input("Value", key=f"rule_val_{i}")
                    rule = {'condition': condition, 'value': value}
            
            with col3:
                category = st.text_input("Category", key=f"rule_cat_{i}")
                rule['category'] = category
            
            rules.append(rule)
        
        params['rules'] = rules
        params['default_category'] = st.text_input("Default Category", value="Other", key="default_cat")
    
    elif relationship_type == 'Category Averages':
        numeric_cols = [col for col in st.session_state.generator.data.columns 
                       if st.session_state.generator.data[col].dtype in ['int64', 'float64']]
        params['numeric_column'] = st.selectbox(
            "Numeric Column for Averages",
            options=numeric_cols,
            key="cat_avg_col"
        )
    
    elif relationship_type == 'Category Counts':
        st.write("This will count occurrences of each category in the source column.")
        # No additional parameters needed
    
    elif relationship_type == 'Encoded Values':
        st.write("This will perform simple label encoding (0, 1, 2, ...) for categories.")
        # No additional parameters needed
    
    elif relationship_type == 'Custom Assignments':
        st.write("**Custom Value Assignments**")
        st.write("Assign numeric values to categories:")
        
        # Get unique values from source column
        source_values = st.session_state.generator.data[source_column].unique()
        assignments = {}
        
        for value in source_values[:10]:  # Limit to first 10 values
            numeric_value = st.number_input(
                f"Assign value to '{value}':",
                value=0.0,
                key=f"assign_{value}"
            )
            assignments[str(value)] = numeric_value
        
        params['assignments'] = assignments
        params['default_value'] = st.number_input("Default value for unassigned categories:", value=0.0, key="assign_default")
    
    elif relationship_type == 'Logical Operations':
        col1, col2 = st.columns(2)
        with col1:
            params['operation'] = st.selectbox(
                "Logical Operation",
                ['AND', 'OR', 'NOT', 'XOR'],
                key="logical_op"
            )
        with col2:
            if params['operation'] != 'NOT':
                params['second_column'] = st.selectbox(
                    "Second Column",
                    options=[col for col in st.session_state.generator.data.columns if col != source_column],
                    key="logical_second"
                )
    
    elif relationship_type == 'Conditional Boolean':
        col1, col2 = st.columns(2)
        with col1:
            params['condition'] = st.selectbox(
                "Condition",
                ['greater_than', 'less_than', 'equal_to'],
                key="bool_cond"
            )
        with col2:
            params['threshold'] = st.number_input("Threshold", value=0.0, key="bool_thresh")
    
    elif relationship_type == 'Threshold-based Boolean':
        col1, col2 = st.columns(2)
        with col1:
            params['lower_threshold'] = st.number_input("Lower Threshold", value=0.0, key="bool_lower")
        with col2:
            params['upper_threshold'] = st.number_input("Upper Threshold", value=1.0, key="bool_upper")
    
    else:
        st.warning(f"Parameters for '{relationship_type}' not yet implemented")
    
    # Add noise option for numeric targets
    if target_type == 'numeric':
        st.write("**Additional Options**")
        params['noise_level'] = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            help="Add random noise to the relationship"
        )
    
    return params

def apply_relationship(interrelations: ColumnInterrelations, source_col: str, target_col: str,
                      relationship_type: str, params: Dict, source_type: str, target_type: str):
    """Apply the selected relationship."""
    if source_type == 'numeric' and target_type == 'numeric':
        interrelations.apply_numeric_numeric_relationship(source_col, target_col, relationship_type, params)
    elif source_type == 'categorical' and target_type == 'categorical':
        interrelations.apply_categorical_categorical_relationship(source_col, target_col, relationship_type, params)
    elif source_type == 'numeric' and target_type == 'categorical':
        interrelations.apply_numeric_categorical_relationship(source_col, target_col, relationship_type, params)
    elif source_type == 'categorical' and target_type == 'numeric':
        interrelations.apply_categorical_numeric_relationship(source_col, target_col, relationship_type, params)
    else:
        st.warning(f"Relationship type {relationship_type} not yet implemented for {source_type} → {target_type}") 