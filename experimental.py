import streamlit as st
import pandas as pd
import plotly.express as px
from data_generator import DataGenerator

def show_experimental_page():
    st.title("Experimental Features")
    st.write("This page contains experimental features for generating related columns.")

    if 'generator' not in st.session_state:
        st.error("Please generate some data in the main page first.")
        return

    if st.session_state.generator.data.empty:
        st.error("No data available. Please add some columns in the main page first.")
        return

    # History controls
    st.header("History Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("↩️ Undo Last Change", disabled=not st.session_state.generator.can_undo()):
            if st.session_state.generator.undo():
                st.success("Undid last change")
                st.rerun()
            else:
                st.error("Nothing to undo")
    
    with col2:
        if st.button("↪️ Redo Last Undo", disabled=not st.session_state.generator.can_redo()):
            if st.session_state.generator.redo():
                st.success("Redid last undone change")
                st.rerun()
            else:
                st.error("Nothing to redo")
    
    with col3:
        if st.button("🔄 Reset to Initial State", disabled=not st.session_state.generator.can_undo()):
            st.session_state.generator.reset_to_initial()
            st.success("Reset to initial state")
            st.rerun()

    st.header("Column Relationships")
    
    # Select columns for relationship
    col1, col2 = st.columns(2)
    
    with col1:
        source_column = st.selectbox(
            "Source Column",
            options=list(st.session_state.generator.data.columns)
        )
        
        target_column = st.selectbox(
            "Target Column",
            options=list(st.session_state.generator.data.columns),
            index=1 if len(st.session_state.generator.data.columns) > 1 else 0
        )

    with col2:
        relationship_type = st.selectbox(
            "Relationship Type",
            ["linear", "polynomial", "exponential", "logarithmic", "trigonometric", "conditional", "inverse"]
        )

        st.subheader("Relationship Parameters")
        params = {}
        
        if relationship_type == "linear":
            params["slope"] = st.number_input("Slope", value=1.0)
            params["intercept"] = st.number_input("Intercept", value=0.0)
        elif relationship_type == "polynomial":
            params["a"] = st.number_input("Quadratic Coefficient (a)", value=1.0)
            params["b"] = st.number_input("Linear Coefficient (b)", value=0.0)
            params["c"] = st.number_input("Constant (c)", value=0.0)
        elif relationship_type == "exponential":
            params["a"] = st.number_input("Base Coefficient (a)", value=1.0)
            params["b"] = st.number_input("Exponent Coefficient (b)", value=1.0)
        elif relationship_type == "logarithmic":
            params["a"] = st.number_input("Coefficient (a)", value=1.0)
            params["b"] = st.number_input("Constant (b)", value=0.0)
            params["offset"] = st.number_input("Offset (to avoid log(0))", value=1.0, min_value=0.1)
        elif relationship_type == "trigonometric":
            params["function"] = st.selectbox("Function", ["sin", "cos", "tan"])
            params["a"] = st.number_input("Amplitude (a)", value=1.0)
            params["b"] = st.number_input("Frequency (b)", value=1.0)
            params["c"] = st.number_input("Phase Shift (c)", value=0.0)
        elif relationship_type == "conditional":
            params["threshold"] = st.number_input("Threshold", value=0.0)
            params["low_value"] = st.number_input("Low Value", value=0.0)
            params["high_value"] = st.number_input("High Value", value=1.0)
        elif relationship_type == "inverse":
            params["a"] = st.number_input("Coefficient (a)", value=1.0)
            params["b"] = st.number_input("Constant (b)", value=0.0)
            params["offset"] = st.number_input("Offset (to avoid division by zero)", value=1e-8, min_value=1e-10)
        
        params["noise"] = st.slider("Noise Level", 0.0, 1.0, 0.1)

    if st.button("Apply Relationship"):
        try:
            st.session_state.generator.add_column_relationship(
                target_column=target_column,
                source_column=source_column,
                relationship_type=relationship_type,
                parameters=params
            )
            st.success("Relationship applied successfully!")
        except Exception as e:
            st.error(f"Error applying relationship: {str(e)}")

    # Display current relationships
    st.header("Current Relationships")
    relationships = st.session_state.generator.get_relationships()
    if relationships:
        for target, rel in relationships.items():
            with st.expander(f"{target} ← {rel['source_column']} ({rel['relationship_type']})"):
                st.write("Parameters:", rel['parameters'])
                if st.button(f"Remove Relationship for {target}"):
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

    # Visualization
    if source_column and target_column:
        st.header("Relationship Visualization")
        fig = px.scatter(
            st.session_state.generator.data,
            x=source_column,
            y=target_column,
            title=f"Relationship between {source_column} and {target_column}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show correlation
        correlation = st.session_state.generator.data[[source_column, target_column]].corr().iloc[0,1]
        st.write(f"Correlation coefficient: {correlation:.3f}")

    # History information
    st.sidebar.header("History Information")
    st.sidebar.write(f"Number of changes: {len(st.session_state.generator.history)}")
    st.sidebar.write(f"Current position: {st.session_state.generator.current_history_index + 1}")
    if st.session_state.generator.can_undo():
        st.sidebar.write("↩️ Undo available")
    if st.session_state.generator.can_redo():
        st.sidebar.write("↪️ Redo available") 