"""
Streamlit Dashboard for Customer Churn Prediction System.
Displays churn predictions, metrics, and SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config
from services.prediction import ChurnPredictor, CustomerRecord
from services.model_repository import ModelRepository
from services.explainability import SHAPExplainer
from services.monitoring import ModelEvaluator, PredictionStore
from services.preprocessing import DataTransformer


# Page configuration
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_data(model_version=None):
    """
    Load model, data, and initialize predictor.
    Cached to avoid reloading on every interaction.
    """
    # Initialize repository
    repository = ModelRepository()
    
    # Load model
    if model_version:
        model, transformer = repository.load(model_version)
    else:
        model, transformer = repository.load_deployed()
        model_version = repository.get_deployed_version()
        if model_version is None:
            model_version = repository.get_latest_version()
    
    # Load test data for predictions
    test_data_path = config.TEST_DATA_PATH
    if test_data_path.exists():
        df = pd.read_csv(test_data_path)
    else:
        df = None
    
    # Load background data for SHAP (use subset of training data)
    training_data_path = config.TRAINING_DATA_PATH
    background_data = None
    if training_data_path.exists():
        train_df = pd.read_csv(training_data_path)
        if 'churn' in train_df.columns:
            X_train = train_df.drop('churn', axis=1)
            # Apply preprocessing
            from services.preprocessing import FeatureEngineer
            feature_engineer = FeatureEngineer()
            feature_engineer.fit(X_train)
            X_train_eng = feature_engineer.transform(X_train)
            X_train_transformed = transformer.transform(X_train_eng)
            # Use subset for SHAP background
            background_data = X_train_transformed[:min(config.SHAP_BACKGROUND_SAMPLES, len(X_train_transformed))]
    
    # Initialize predictor with explanations if background data available
    enable_explanations = background_data is not None
    predictor = ChurnPredictor(
        model_version=model_version,
        repository=repository,
        enable_explanations=enable_explanations,
        background_data=background_data
    )
    
    # Get metadata
    metadata = repository.get_metadata(model_version)
    
    return predictor, df, metadata, repository


def main():
    """Main dashboard application."""
    
    st.title("📊 " + config.DASHBOARD_TITLE)
    st.markdown("---")
    
    # Sidebar for model version selection
    st.sidebar.header("⚙️ Settings")
    
    # Initialize repository to get versions
    repository = ModelRepository()
    versions = repository.list_versions()
    
    if not versions:
        st.error("No models found in repository. Please train a model first.")
        return
    
    # Model version selector
    version_options = [v.version for v in versions]
    deployed_version = repository.get_deployed_version()
    
    if deployed_version and deployed_version in version_options:
        default_index = version_options.index(deployed_version)
    else:
        default_index = 0
    
    selected_version = st.sidebar.selectbox(
        "Select Model Version",
        options=version_options,
        index=default_index,
        help="Choose which model version to use for predictions"
    )
    
    # Load model and data
    try:
        predictor, test_df, metadata, repo = load_model_and_data(selected_version)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Display model metadata in sidebar
    st.sidebar.markdown("### 📋 Model Information")
    st.sidebar.text(f"Version: {metadata.get('version', 'N/A')}")
    st.sidebar.text(f"Features: {metadata.get('n_features', 'N/A')}")
    
    if 'metrics' in metadata:
        metrics = metadata['metrics']
        st.sidebar.text(f"Recall: {metrics.get('recall', 0):.3f}")
        st.sidebar.text(f"Precision: {metrics.get('precision', 0):.3f}")
        st.sidebar.text(f"F1-Score: {metrics.get('f1_score', 0):.3f}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "👤 Customer Details", "📊 Statistics"])
    
    # Tab 1: Main Page (Overview)
    with tab1:
        st.header("Overview")
        
        if test_df is None:
            st.warning("No test data available. Please add test data to generate predictions.")
            return
        
        # Generate predictions for all customers
        with st.spinner("Generating predictions..."):
            # Prepare customer records
            customers = []
            for idx, row in test_df.iterrows():
                customer_id = f"CUST_{idx:04d}"
                features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
                customers.append({
                    'customer_id': customer_id,
                    **features
                })
            
            # Get predictions
            predictions = predictor.predict_batch(customers, include_explanations=False)
        
        # Extract prediction data
        pred_df = pd.DataFrame([
            {
                'customer_id': p.customer_id,
                'churn_probability': p.churn_probability,
                'is_high_risk': p.is_high_risk
            }
            for p in predictions
        ])
        
        # Display high-risk account count
        high_risk_count = pred_df['is_high_risk'].sum()
        total_count = len(pred_df)
        high_risk_percentage = (high_risk_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🚨 High-Risk Accounts",
                value=f"{high_risk_count}",
                delta=f"{high_risk_percentage:.1f}% of total"
            )
        
        with col2:
            avg_churn_prob = pred_df['churn_probability'].mean()
            st.metric(
                label="📊 Average Churn Probability",
                value=f"{avg_churn_prob:.3f}"
            )
        
        with col3:
            st.metric(
                label="👥 Total Customers",
                value=f"{total_count}"
            )
        
        st.markdown("---")
        
        # Churn probability distribution histogram
        st.subheader("Churn Probability Distribution")
        
        fig_hist = px.histogram(
            pred_df,
            x='churn_probability',
            nbins=30,
            title="Distribution of Churn Probabilities",
            labels={'churn_probability': 'Churn Probability', 'count': 'Number of Customers'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add vertical line for high-risk threshold
        fig_hist.add_vline(
            x=config.HIGH_RISK_THRESHOLD,
            line_dash="dash",
            line_color="red",
            annotation_text=f"High-Risk Threshold ({config.HIGH_RISK_THRESHOLD})",
            annotation_position="top"
        )
        
        fig_hist.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        
        if 'metrics' in metadata:
            metrics = metadata['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                recall = metrics.get('recall', 0)
                recall_delta = "✅ Meets threshold" if recall >= config.MIN_RECALL_THRESHOLD else "⚠️ Below threshold"
                st.metric(
                    label="Recall",
                    value=f"{recall:.3f}",
                    delta=recall_delta
                )
            
            with col2:
                st.metric(
                    label="Precision",
                    value=f"{metrics.get('precision', 0):.3f}"
                )
            
            with col3:
                st.metric(
                    label="F1-Score",
                    value=f"{metrics.get('f1_score', 0):.3f}"
                )
            
            with col4:
                st.metric(
                    label="Accuracy",
                    value=f"{metrics.get('accuracy', 0):.3f}"
                )
        else:
            st.info("No performance metrics available for this model version.")
        
        # Show prediction table
        st.markdown("---")
        st.subheader("Recent Predictions")
        
        # Add actual churn if available
        if 'churn' in test_df.columns:
            pred_df['actual_churn'] = test_df['churn'].values
        
        # Display table
        display_df = pred_df.head(20).copy()
        display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x:.3f}")
        display_df['is_high_risk'] = display_df['is_high_risk'].apply(lambda x: "🚨 Yes" if x else "✅ No")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # Tab 2: Customer Detail View
    with tab2:
        st.header("Customer Details")
        
        if test_df is None:
            st.warning("No test data available.")
            return
        
        # Customer selection dropdown
        customer_ids = [f"CUST_{idx:04d}" for idx in range(len(test_df))]
        selected_customer_id = st.selectbox(
            "Select Customer",
            options=customer_ids,
            help="Choose a customer to view detailed prediction and explanation"
        )
        
        # Get customer index
        customer_idx = int(selected_customer_id.split('_')[1])
        customer_row = test_df.iloc[customer_idx]
        
        # Prepare customer record
        features = customer_row.drop('churn').to_dict() if 'churn' in customer_row else customer_row.to_dict()
        customer = {
            'customer_id': selected_customer_id,
            **features
        }
        
        # Generate prediction with explanation
        with st.spinner("Generating prediction and explanation..."):
            try:
                result = predictor.predict_single(customer, include_explanations=True)
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                result = predictor.predict_single(customer, include_explanations=False)
        
        # Display customer information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            st.text(f"Customer ID: {selected_customer_id}")
            st.text(f"Tenure: {features.get('tenure_months', 'N/A')} months")
            st.text(f"Monthly Charges: ${features.get('monthly_charges', 0):.2f}")
            st.text(f"Total Charges: ${features.get('total_charges', 0):.2f}")
            st.text(f"Contract: {features.get('contract_type', 'N/A')}")
            st.text(f"Payment Method: {features.get('payment_method', 'N/A')}")
            st.text(f"Internet Service: {features.get('internet_service', 'N/A')}")
        
        with col2:
            st.subheader("Prediction")
            
            # Churn probability gauge
            churn_prob = result.churn_probability
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if result.is_high_risk else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': config.HIGH_RISK_THRESHOLD * 100
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if result.is_high_risk:
                st.error(f"🚨 HIGH RISK - Churn Probability: {churn_prob:.3f}")
            else:
                st.success(f"✅ LOW RISK - Churn Probability: {churn_prob:.3f}")
        
        # Display top contributing features
        st.markdown("---")
        st.subheader("Top Contributing Features")
        
        if result.top_features:
            # Create DataFrame for top features
            top_features_df = pd.DataFrame(
                result.top_features,
                columns=['Feature', 'SHAP Value']
            )
            
            # Create horizontal bar chart
            fig_features = px.bar(
                top_features_df,
                x='SHAP Value',
                y='Feature',
                orientation='h',
                title="Top 5 Features Contributing to Churn Prediction",
                labels={'SHAP Value': 'SHAP Value (Impact on Prediction)', 'Feature': 'Feature Name'},
                color='SHAP Value',
                color_continuous_scale=['green', 'yellow', 'red'],
                color_continuous_midpoint=0
            )
            
            fig_features.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Explanation text
            st.markdown("### 📝 Interpretation")
            st.markdown("""
            - **Positive SHAP values** (red) indicate features that *increase* churn risk
            - **Negative SHAP values** (green) indicate features that *decrease* churn risk
            - Larger absolute values indicate stronger influence on the prediction
            """)
            
            # Display feature values
            st.markdown("### Feature Values")
            feature_values = []
            for feature_name, shap_value in result.top_features:
                # Try to get the original feature value
                if feature_name in features:
                    value = features[feature_name]
                    impact = "Increases" if shap_value > 0 else "Decreases"
                    feature_values.append({
                        'Feature': feature_name,
                        'Value': value,
                        'SHAP Value': f"{shap_value:.4f}",
                        'Impact': f"{impact} churn risk"
                    })
            
            if feature_values:
                st.table(pd.DataFrame(feature_values))
        else:
            st.info("SHAP explanations not available. Enable explanations when initializing the predictor.")
    
    # Tab 3: Aggregated Statistics
    with tab3:
        st.header("Aggregated Statistics")
        
        if test_df is None:
            st.warning("No test data available.")
            return
        
        # Generate predictions if not already done
        if 'pred_df' not in locals():
            with st.spinner("Generating predictions..."):
                customers = []
                for idx, row in test_df.iterrows():
                    customer_id = f"CUST_{idx:04d}"
                    features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
                    customers.append({
                        'customer_id': customer_id,
                        **features
                    })
                
                predictions = predictor.predict_batch(customers, include_explanations=True)
        
        # Compute aggregate statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_prob = np.mean([p.churn_probability for p in predictions])
            st.metric("Average Churn Probability", f"{avg_prob:.3f}")
        
        with col2:
            median_prob = np.median([p.churn_probability for p in predictions])
            st.metric("Median Churn Probability", f"{median_prob:.3f}")
        
        with col3:
            std_prob = np.std([p.churn_probability for p in predictions])
            st.metric("Std Dev Churn Probability", f"{std_prob:.3f}")
        
        st.markdown("---")
        
        # Feature importance trends
        st.subheader("Feature Importance Trends")
        
        if predictions[0].top_features:
            # Aggregate feature importance across all customers
            feature_importance = {}
            
            for pred in predictions:
                if pred.top_features:
                    for feature_name, shap_value in pred.top_features:
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = []
                        feature_importance[feature_name].append(abs(shap_value))
            
            # Compute average absolute SHAP value for each feature
            avg_importance = {
                feature: np.mean(values)
                for feature, values in feature_importance.items()
            }
            
            # Sort by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            importance_df = pd.DataFrame(
                sorted_features[:10],  # Top 10 features
                columns=['Feature', 'Average Absolute SHAP Value']
            )
            
            fig_importance = px.bar(
                importance_df,
                x='Average Absolute SHAP Value',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features (Across All Customers)",
                labels={'Average Absolute SHAP Value': 'Average |SHAP Value|', 'Feature': 'Feature Name'}
            )
            
            fig_importance.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
        
        st.markdown("---")
        
        # Retention metrics summary
        st.subheader("Retention Metrics Summary")
        
        if 'churn' in test_df.columns:
            # Compute retention metrics
            actual_churn = test_df['churn'].values
            predicted_churn = [p.is_high_risk for p in predictions]
            
            # True positives, false positives, etc.
            tp = sum((a == 1 and p) for a, p in zip(actual_churn, predicted_churn))
            fp = sum((a == 0 and p) for a, p in zip(actual_churn, predicted_churn))
            tn = sum((a == 0 and not p) for a, p in zip(actual_churn, predicted_churn))
            fn = sum((a == 1 and not p) for a, p in zip(actual_churn, predicted_churn))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("True Positives", tp)
            with col2:
                st.metric("False Positives", fp)
            with col3:
                st.metric("True Negatives", tn)
            with col4:
                st.metric("False Negatives", fn)
            
            # Confusion matrix visualization
            st.markdown("### Confusion Matrix")
            
            confusion_data = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                columns=['Predicted: No Churn', 'Predicted: Churn'],
                index=['Actual: No Churn', 'Actual: Churn']
            )
            
            fig_confusion = px.imshow(
                confusion_data,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                color_continuous_scale='Blues'
            )
            
            fig_confusion.update_layout(height=400)
            st.plotly_chart(fig_confusion, use_container_width=True)
            
            # Retention rate
            st.markdown("### Retention Metrics")
            
            actual_retention_rate = (1 - actual_churn.mean()) * 100
            predicted_high_risk_rate = (sum(predicted_churn) / len(predicted_churn)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Actual Retention Rate", f"{actual_retention_rate:.1f}%")
            with col2:
                st.metric("Predicted High-Risk Rate", f"{predicted_high_risk_rate:.1f}%")
        else:
            st.info("Actual churn labels not available in test data.")


if __name__ == "__main__":
    main()
