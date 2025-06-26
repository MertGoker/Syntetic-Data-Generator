import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import io

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        
    def load_data(self, uploaded_file) -> bool:
        """Load data from uploaded file."""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return False
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_basic_info(self) -> Dict:
        """Get basic information about the dataset."""
        if self.data is None:
            return {}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        # Add column type classification
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.data.select_dtypes(include=['datetime']).columns.tolist()
        
        info['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
        
        return info
    
    def analyze_numeric_column(self, column: str) -> Dict:
        """Analyze a numeric column."""
        if column not in self.data.columns:
            return {}
        
        col_data = self.data[column].dropna()
        
        if len(col_data) == 0:
            return {'error': 'No valid data in column'}
        
        analysis = {
            'count': len(col_data),
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'q25': col_data.quantile(0.25),
            'q75': col_data.quantile(0.75),
            'skewness': col_data.skew(),
            'kurtosis': col_data.kurtosis(),
            'missing_count': self.data[column].isnull().sum(),
            'missing_percentage': (self.data[column].isnull().sum() / len(self.data)) * 100
        }
        
        # Outlier detection using IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        analysis['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(col_data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return analysis
    
    def analyze_categorical_column(self, column: str) -> Dict:
        """Analyze a categorical column."""
        if column not in self.data.columns:
            return {}
        
        col_data = self.data[column]
        
        analysis = {
            'count': len(col_data),
            'unique_count': col_data.nunique(),
            'missing_count': col_data.isnull().sum(),
            'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
            'value_counts': col_data.value_counts().to_dict(),
            'value_counts_percentage': (col_data.value_counts() / len(col_data) * 100).to_dict()
        }
        
        # Cardinality analysis
        if analysis['unique_count'] <= 10:
            analysis['cardinality'] = 'Low'
        elif analysis['unique_count'] <= 50:
            analysis['cardinality'] = 'Medium'
        else:
            analysis['cardinality'] = 'High'
        
        return analysis
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for numeric columns."""
        if self.data is None:
            return pd.DataFrame()
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.corr()
    
    def detect_relationships(self) -> List[Dict]:
        """Detect potential relationships between columns."""
        if self.data is None:
            return []
        
        relationships = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Calculate correlation
                correlation = self.data[col1].corr(self.data[col2])
                
                if abs(correlation) > 0.3:  # Threshold for meaningful correlation
                    relationships.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': correlation,
                        'strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'
                    })
        
        # Sort by absolute correlation
        relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return relationships
    
    def generate_plots(self, column: str) -> Dict:
        """Generate plots for a specific column."""
        if column not in self.data.columns:
            return {}
        
        plots = {}
        col_data = self.data[column].dropna()
        
        if len(col_data) == 0:
            return {'error': 'No valid data for plotting'}
        
        # Determine column type
        if self.data[column].dtype in ['object', 'category']:
            # Categorical column
            value_counts = self.data[column].value_counts()
            
            # Bar plot
            fig_bar = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Value Counts for {column}",
                labels={'x': column, 'y': 'Count'}
            )
            plots['bar_plot'] = fig_bar
            
            # Pie chart (if not too many categories)
            if len(value_counts) <= 10:
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {column}"
                )
                plots['pie_plot'] = fig_pie
                
        else:
            # Numeric column
            # Histogram
            fig_hist = px.histogram(
                col_data,
                x=column,
                title=f"Distribution of {column}",
                nbins=min(50, len(col_data) // 10)
            )
            plots['histogram'] = fig_hist
            
            # Box plot
            fig_box = px.box(
                col_data,
                y=column,
                title=f"Box Plot of {column}"
            )
            plots['box_plot'] = fig_box
            
            # Q-Q plot for normality test
            if len(col_data) > 3:
                qq_data = stats.probplot(col_data, dist="norm")
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='Data'
                ))
                fig_qq.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                    mode='lines',
                    name='Normal Distribution'
                ))
                fig_qq.update_layout(
                    title=f"Q-Q Plot for {column}",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                plots['qq_plot'] = fig_qq
        
        return plots
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if self.data is None:
            return "No data loaded."
        
        info = self.get_basic_info()
        report = f"""
# Dataset Summary Report

## Basic Information
- **Shape**: {info['shape'][0]} rows × {info['shape'][1]} columns
- **Memory Usage**: {info['memory_usage'] / 1024:.2f} KB
- **Duplicate Rows**: {info['duplicate_rows']}

## Column Types
- **Numeric Columns**: {len(info['column_types']['numeric'])} ({', '.join(info['column_types']['numeric']) if info['column_types']['numeric'] else 'None'})
- **Categorical Columns**: {len(info['column_types']['categorical'])} ({', '.join(info['column_types']['categorical']) if info['column_types']['categorical'] else 'None'})
- **Datetime Columns**: {len(info['column_types']['datetime'])} ({', '.join(info['column_types']['datetime']) if info['column_types']['datetime'] else 'None'})

## Missing Values
"""
        
        for col, missing in info['missing_values'].items():
            if missing > 0:
                percentage = (missing / info['shape'][0]) * 100
                report += f"- **{col}**: {missing} ({percentage:.1f}%)\n"
        
        if not any(info['missing_values'].values()):
            report += "- No missing values found\n"
        
        # Add correlation insights
        if len(info['column_types']['numeric']) > 1:
            relationships = self.detect_relationships()
            if relationships:
                report += "\n## Strongest Correlations\n"
                for rel in relationships[:5]:  # Top 5
                    report += f"- **{rel['column1']}** ↔ **{rel['column2']}**: {rel['correlation']:.3f} ({rel['strength']})\n"
        
        return report

def show_data_analysis_page():
    """Show the data analysis page."""
    st.title("📊 Data Analysis")
    st.write("Upload your CSV file to analyze its structure and content.")
    
    # Initialize analyzer in session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file to analyze"
    )
    
    if uploaded_file is not None:
        # Load data
        if st.session_state.analyzer.load_data(uploaded_file):
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            # Get basic info
            info = st.session_state.analyzer.get_basic_info()
            
            # Display basic information
            st.header("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", info['shape'][0])
            with col2:
                st.metric("Columns", info['shape'][1])
            with col3:
                st.metric("Memory Usage", f"{info['memory_usage'] / 1024:.1f} KB")
            with col4:
                st.metric("Duplicate Rows", info['duplicate_rows'])
            
            # Data preview
            st.subheader("📄 Data Preview")
            st.dataframe(st.session_state.analyzer.data.head(10))
            
            # Column information
            st.subheader("📊 Column Information")
            col_info_df = pd.DataFrame({
                'Column': info['columns'],
                'Data Type': [str(info['dtypes'][col]) for col in info['columns']],
                'Missing Values': [info['missing_values'][col] for col in info['columns']],
                'Missing %': [f"{(info['missing_values'][col] / info['shape'][0]) * 100:.1f}%" for col in info['columns']]
            })
            st.dataframe(col_info_df, use_container_width=True)
            
            # Detailed analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Column Analysis", "🔗 Correlations", "📋 Summary Report", "📊 Visualizations"])
            
            with tab1:
                st.subheader("Individual Column Analysis")
                
                # Column selector
                selected_column = st.selectbox(
                    "Select a column to analyze",
                    options=info['columns']
                )
                
                if selected_column:
                    if selected_column in info['column_types']['numeric']:
                        # Numeric column analysis
                        analysis = st.session_state.analyzer.analyze_numeric_column(selected_column)
                        
                        if 'error' not in analysis:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Basic Statistics**")
                                stats_df = pd.DataFrame({
                                    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                                    'Value': [
                                        analysis['count'],
                                        f"{analysis['mean']:.4f}",
                                        f"{analysis['median']:.4f}",
                                        f"{analysis['std']:.4f}",
                                        f"{analysis['min']:.4f}",
                                        f"{analysis['max']:.4f}",
                                        f"{analysis['q25']:.4f}",
                                        f"{analysis['q75']:.4f}"
                                    ]
                                })
                                st.dataframe(stats_df, use_container_width=True)
                            
                            with col2:
                                st.write("**Distribution Properties**")
                                dist_df = pd.DataFrame({
                                    'Property': ['Skewness', 'Kurtosis', 'Missing Count', 'Missing %'],
                                    'Value': [
                                        f"{analysis['skewness']:.4f}",
                                        f"{analysis['kurtosis']:.4f}",
                                        analysis['missing_count'],
                                        f"{analysis['missing_percentage']:.2f}%"
                                    ]
                                })
                                st.dataframe(dist_df, use_container_width=True)
                            
                            # Outlier information
                            st.write("**Outlier Analysis (IQR Method)**")
                            outlier_df = pd.DataFrame({
                                'Metric': ['Outlier Count', 'Outlier %', 'Lower Bound', 'Upper Bound'],
                                'Value': [
                                    analysis['outliers']['count'],
                                    f"{analysis['outliers']['percentage']:.2f}%",
                                    f"{analysis['outliers']['lower_bound']:.4f}",
                                    f"{analysis['outliers']['upper_bound']:.4f}"
                                ]
                            })
                            st.dataframe(outlier_df, use_container_width=True)
                        else:
                            st.error(analysis['error'])
                    
                    else:
                        # Categorical column analysis
                        analysis = st.session_state.analyzer.analyze_categorical_column(selected_column)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Basic Information**")
                            basic_df = pd.DataFrame({
                                'Metric': ['Count', 'Unique Values', 'Missing Count', 'Missing %', 'Cardinality'],
                                'Value': [
                                    analysis['count'],
                                    analysis['unique_count'],
                                    analysis['missing_count'],
                                    f"{analysis['missing_percentage']:.2f}%",
                                    analysis['cardinality']
                                ]
                            })
                            st.dataframe(basic_df, use_container_width=True)
                        
                        with col2:
                            st.write("**Value Counts**")
                            if len(analysis['value_counts']) <= 20:
                                value_counts_df = pd.DataFrame({
                                    'Value': list(analysis['value_counts'].keys()),
                                    'Count': list(analysis['value_counts'].values()),
                                    'Percentage': [f"{analysis['value_counts_percentage'][k]:.2f}%" for k in analysis['value_counts'].keys()]
                                })
                                st.dataframe(value_counts_df, use_container_width=True)
                            else:
                                st.write(f"Too many unique values ({analysis['unique_count']}) to display all. Showing top 10:")
                                top_values = dict(list(analysis['value_counts'].items())[:10])
                                value_counts_df = pd.DataFrame({
                                    'Value': list(top_values.keys()),
                                    'Count': list(top_values.values()),
                                    'Percentage': [f"{analysis['value_counts_percentage'][k]:.2f}%" for k in top_values.keys()]
                                })
                                st.dataframe(value_counts_df, use_container_width=True)
            
            with tab2:
                st.subheader("Correlation Analysis")
                
                # Correlation matrix
                corr_matrix = st.session_state.analyzer.get_correlation_matrix()
                
                if not corr_matrix.empty:
                    # Heatmap
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Matrix Heatmap",
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strongest correlations
                    relationships = st.session_state.analyzer.detect_relationships()
                    if relationships:
                        st.write("**Strongest Correlations**")
                        rel_df = pd.DataFrame(relationships)
                        st.dataframe(rel_df, use_container_width=True)
                    else:
                        st.info("No strong correlations found (|r| > 0.3)")
                else:
                    st.info("No numeric columns available for correlation analysis")
            
            with tab3:
                st.subheader("Summary Report")
                report = st.session_state.analyzer.generate_summary_report()
                st.markdown(report)

                st.download_button(
                    label="📥 Download Report as Markdown",
                    data=report,
                    file_name="data_analysis_report.md",
                    mime="text/markdown"
                )
            
            with tab4:
                st.subheader("Column Visualizations")
                
                # Column selector for plots
                plot_column = st.selectbox(
                    "Select a column to visualize",
                    options=info['columns'],
                    key="plot_column"
                )
                
                if plot_column:
                    plots = st.session_state.analyzer.generate_plots(plot_column)
                    
                    if 'error' not in plots:
                        for plot_name, fig in plots.items():
                            st.write(f"**{plot_name.replace('_', ' ').title()}**")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(plots['error'])
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis") 