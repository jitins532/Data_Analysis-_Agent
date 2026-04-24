import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
import hashlib


# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, mean_absolute_error
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

# -------------------- Authentication Functions --------------------
def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify a hashed password."""
    return hash_password(password) == hashed_password

def init_authentication():
    """Initialize authentication system."""
    if 'users' not in st.session_state:
        # Default users (username: password)
        st.session_state.users = {
            'admin': hash_password('admin123'),
            'user': hash_password('user123'),
            'Jitin': hash_password('Jitin@123')
        }
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

def login_form():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="login-header">🔐 AutoML Login</h2>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        remember_me = st.checkbox("Remember me")
        
        submitted = st.form_submit_button("🚀 Login")
        
        if submitted:
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.login_attempts = 0
                st.success(f"Welcome, {username}! 👋")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining_attempts = 5 - st.session_state.login_attempts
                if remaining_attempts > 0:
                    st.error(f"❌ Invalid credentials! {remaining_attempts} attempts remaining.")
                else:
                    st.error("🚫 Too many failed attempts! Please refresh the page.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def authenticate_user(username, password):
    """Authenticate user credentials."""
    if username in st.session_state.users:
        return verify_password(password, st.session_state.users[username])
    return False

def logout():
    """Logout user and reset session."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.login_attempts = 0
    # Clear all ML-related session states
    ml_states = ['model_trained', 'model', 'X_train', 'X_test', 'y_train', 'y_test', 
                 'label_encoders', 'scaler', 'imputer', 'problem_type', 'target_col', 
                 'feature_names', 'df', 'fitted_imputer', 'chat_history']
    for state in ml_states:
        if state in st.session_state:
            del st.session_state[state]

def user_management_section():
    """User management section (admin only)."""
    if st.session_state.current_user == 'admin':
        st.sidebar.markdown("---")
        st.sidebar.subheader("👥 User Management (Admin)")
        
        with st.sidebar.expander("Manage Users"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add User"):
                    if new_username and new_password:
                        if new_username not in st.session_state.users:
                            st.session_state.users[new_username] = hash_password(new_password)
                            st.success(f"User '{new_username}' added successfully!")
                        else:
                            st.error("Username already exists!")
            
            with col2:
                if st.button("Remove User"):
                    if new_username in st.session_state.users and new_username != 'admin':
                        del st.session_state.users[new_username]
                        st.success(f"User '{new_username}' removed successfully!")
                    else:
                        st.error("Cannot remove admin or user doesn't exist!")
            
            # Display current users
            st.write("**Current Users:**")
            for user in st.session_state.users:
                st.write(f"- {user}")

# -------------------- Enhanced AI Assistant Functions --------------------
def init_chat():
    """Initialize chat history."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm your AutoML Assistant. I can help you with data analysis, model training, and interpretation. How can I assist you today?"}
        ]

def add_message(role, content):
    """Add a message to chat history."""
    st.session_state.chat_history.append({"role": role, "content": content})

def analyze_dataset(df):
    """Analyze dataset and return insights."""
    if df is None:
        return "No dataset available."
    
    insights = []
    
    # Basic info
    insights.append(f"**Dataset Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    insights.append(f"**Numeric Columns**: {len(numeric_cols)}")
    insights.append(f"**Categorical Columns**: {len(categorical_cols)}")
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    missing_percent = (missing_total / (df.shape[0] * df.shape[1])) * 100
    insights.append(f"**Missing Values**: {missing_total} ({missing_percent:.2f}%)")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    insights.append(f"**Duplicate Rows**: {duplicates}")
    
    # Basic statistics for numeric columns
    if len(numeric_cols) > 0:
        insights.append("\n**Numeric Columns Summary**:")
        for col in numeric_cols[:5]:  # Show first 5 columns
            col_stats = df[col].describe()
            insights.append(f"- {col}: mean={col_stats['mean']:.2f}, std={col_stats['std']:.2f}, range=[{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
    
    # Categorical columns info
    if len(categorical_cols) > 0:
        insights.append("\n**Categorical Columns (first 3)**:")
        for col in categorical_cols[:3]:
            unique_vals = df[col].nunique()
            top_value = df[col].mode().iloc[0] if not df[col].empty else "N/A"
            insights.append(f"- {col}: {unique_vals} unique values, most frequent: '{top_value}'")
    
    return "\n".join(insights)

def analyze_model_state(app_state):
    """Analyze current model state and return insights."""
    if not app_state.model:
        return "No model has been trained yet."
    
    insights = []
    insights.append(f"**Problem Type**: {app_state.problem_type}")
    insights.append(f"**Target Variable**: {app_state.target_col}")
    insights.append(f"**Features Used**: {len(app_state.feature_names)}")
    
    if app_state.X_train is not None:
        insights.append(f"**Training Samples**: {len(app_state.X_train)}")
        insights.append(f"**Test Samples**: {len(app_state.X_test)}")
    
    # Model specific info
    model_type = type(app_state.model).__name__
    insights.append(f"**Model Algorithm**: {model_type}")
    
    # Feature importance if available
    if hasattr(app_state.model, 'feature_importances_'):
        if len(app_state.feature_names) > 0:
            importances = app_state.model.feature_importances_
            top_features_idx = np.argsort(importances)[-3:][::-1]  # Top 3 features
            insights.append("\n**Top 3 Important Features**:")
            for idx in top_features_idx:
                if idx < len(app_state.feature_names):
                    insights.append(f"- {app_state.feature_names[idx]}: {importances[idx]:.3f}")
    
    return "\n".join(insights)

def get_ai_response(user_message, app_state):
    """Generate intelligent AI response based on user message and app state."""
    user_message_lower = user_message.lower()
    
    # Data analysis queries
    if any(word in user_message_lower for word in ['data', 'dataset', 'overview', 'analyze data']):
        if app_state.df is not None:
            data_insights = analyze_dataset(app_state.df)
            return f"Here's your dataset analysis:\n\n{data_insights}\n\nYou can explore more in the 'Data Overview' section."
        else:
            return "No dataset is currently loaded. Please upload a CSV file in the sidebar to begin analysis."
    
    # Model training queries
    elif any(word in user_message_lower for word in ['train', 'model', 'algorithm', 'build model']):
        if app_state.model is not None:
            model_insights = analyze_model_state(app_state)
            return f"Here's your current model status:\n\n{model_insights}\n\nYou can evaluate the model in the 'Model Evaluation' section."
        else:
            return "No model has been trained yet. Go to the 'Model Training' section to:\n1. Select your target variable\n2. Choose features\n3. Select an algorithm\n4. Train your model"
    
    # Preprocessing queries
    elif any(word in user_message_lower for word in ['preprocess', 'clean', 'missing', 'handle data']):
        if app_state.df is not None:
            missing_info = f"Missing values: {app_state.df.isnull().sum().sum()}\n"
            duplicate_info = f"Duplicate rows: {app_state.df.duplicated().sum()}\n"
            advice = "In the 'Data Preprocessing' section, you can:\n- Handle missing values\n- Remove duplicates\n- Drop columns\n- Create new features"
            return missing_info + duplicate_info + advice
        else:
            return "No data available for preprocessing. Please upload a dataset first."
    
    # Evaluation queries
    elif any(word in user_message_lower for word in ['evaluate', 'performance', 'metrics', 'results']):
        if app_state.model is not None:
            if app_state.X_test is not None:
                try:
                    y_pred = app_state.model.predict(app_state.X_test)
                    if app_state.problem_type == "Classification":
                        accuracy = accuracy_score(app_state.y_test, y_pred)
                        return f"Model evaluation ready! Current accuracy: {accuracy:.3f}\n\nGo to 'Model Evaluation' for detailed metrics, confusion matrix, and classification report."
                    else:
                        r2 = r2_score(app_state.y_test, y_pred)
                        mse = mean_squared_error(app_state.y_test, y_pred)
                        return f"Model evaluation ready! R² Score: {r2:.3f}, MSE: {mse:.3f}\n\nCheck 'Model Evaluation' for comprehensive regression metrics."
                except:
                    return "Model is trained but evaluation data isn't available. Please retrain the model or check the 'Model Evaluation' section."
            else:
                return "Model is trained but test data isn't available. Please retrain the model."
        else:
            return "No model available for evaluation. Please train a model first in the 'Model Training' section."
    
    # Prediction queries
    elif any(word in user_message_lower for word in ['predict', 'forecast', 'make prediction']):
        if app_state.model is not None:
            return f"Ready for predictions! Go to the 'Prediction' section to:\n- Input feature values\n- Get real-time predictions\n- See prediction probabilities (for classification)\n\nCurrent model: {type(app_state.model).__name__} for {app_state.problem_type}"
        else:
            return "No model trained yet. Please train a model first to make predictions."
    
    # Feature selection queries
    elif any(word in user_message_lower for word in ['feature', 'important', 'selection']):
        if app_state.model is not None and hasattr(app_state.model, 'feature_importances_'):
            if len(app_state.feature_names) > 0:
                importances = app_state.model.feature_importances_
                top_idx = np.argsort(importances)[-5:][::-1]  # Top 5 features
                response = "**Top 5 Most Important Features**:\n"
                for i, idx in enumerate(top_idx, 1):
                    if idx < len(app_state.feature_names):
                        response += f"{i}. {app_state.feature_names[idx]}: {importances[idx]:.3f}\n"
                return response
        return "Feature importance is available after training tree-based models (Random Forest, Decision Tree)."
    
    # Algorithm recommendation
    elif any(word in user_message_lower for word in ['which algorithm', 'what model', 'recommend', 'best algorithm']):
        if app_state.df is not None and app_state.target_col:
            target_sample = app_state.df[app_state.target_col].dropna()
            unique_values = len(target_sample.unique())
            
            if unique_values <= 10:
                rec_type = "Classification"
                algorithms = [
                    "Random Forest - Good for most problems, handles non-linearity well",
                    "Logistic Regression - Good for linear relationships, interpretable",
                    "K-Nearest Neighbors - Simple, good for small datasets",
                    "Decision Tree - Very interpretable, good for understanding data"
                ]
            else:
                rec_type = "Regression"
                algorithms = [
                    "Random Forest - Robust, handles non-linearity well",
                    "Linear Regression - Good for linear relationships, interpretable",
                    "Decision Tree - Interpretable, good for understanding patterns"
                ]
            
            response = f"Based on your target variable ({unique_values} unique values), this appears to be a **{rec_type}** problem.\n\n**Recommended Algorithms**:\n"
            for i, algo in enumerate(algorithms, 1):
                response += f"{i}. {algo}\n"
            
            return response
        else:
            return "To recommend algorithms, I need to know your target variable. Please select a target variable in the 'Model Training' section first."
    
    # Data quality issues
    elif any(word in user_message_lower for word in ['problem', 'issue', 'error', 'not working', 'why']):
        issues = []
        if app_state.df is not None:
            if app_state.df.isnull().sum().sum() > 0:
                issues.append("• Missing values in dataset")
            if app_state.df.duplicated().sum() > 0:
                issues.append("• Duplicate rows in dataset")
            if app_state.df.empty:
                issues.append("• Dataset is empty")
        
        if app_state.model is None and 'model' in user_message_lower:
            issues.append("• No model trained yet")
        
        if issues:
            return "I noticed these potential issues:\n" + "\n".join(issues) + "\n\nVisit the relevant sections to address them."
        else:
            return "No major issues detected. If you're experiencing specific problems, please describe them in more detail."
    
    # Next steps guidance
    elif any(word in user_message_lower for word in ['what next', 'next step', 'what should i do']):
        if app_state.df is None:
            return "**Next Step**: Upload a dataset using the file uploader in the sidebar."
        elif app_state.model is None:
            return "**Next Step**: Go to 'Model Training' to build your first machine learning model."
        elif app_state.model is not None:
            return "**Next Step**: Evaluate your model in 'Model Evaluation' or make predictions in 'Prediction' section."
        else:
            return "**Next Step**: Explore your data in 'EDA & Visualization' to gain insights."
    
    # General help
    elif any(word in user_message_lower for word in ['help', 'what can you do', 'assist', 'guide']):
        return """I'm your AutoML Assistant! Here's what I can help you with:

**Data Analysis**:
• Dataset overview and statistics
• Missing value analysis
• Data quality assessment

**Model Training**:
• Algorithm recommendations
• Feature selection guidance
• Hyperparameter advice

**Model Evaluation**:
• Performance interpretation
• Metric explanations
• Result analysis

**Predictions**:
• Prediction guidance
• Confidence scores
• Feature importance

**Troubleshooting**:
• Error diagnosis
• Data issues
• Model problems

Just ask me anything about your machine learning workflow!"""
    
    # Default response with context awareness
    else:
        # Provide context-aware default response
        context = []
        if app_state.df is not None:
            context.append(f"dataset with {app_state.df.shape[0]} rows")
        if app_state.model is not None:
            context.append(f"trained {app_state.problem_type} model")
        
        context_str = " and ".join(context) if context else "the app"
        
        return f"I understand you're asking about: '{user_message}'. \n\nIn the context of your {context_str}, I'd recommend:\n\n• Checking the relevant section for detailed analysis\n• Asking me specific questions about your data or model\n• Using the navigation to explore different features\n\nHow else can I assist you with your machine learning project?"

def ai_assistant_chat(app_state):
    """AI Assistant chat interface."""
    st.markdown("---")
    st.subheader("🤖 AutoML Assistant")
    
    # Initialize chat
    init_chat()
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data or model..."):
        # Add user message to chat history
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data and model..."):
                response = get_ai_response(prompt, app_state)
                st.markdown(response)
        
        # Add assistant response to chat history
        add_message("assistant", response)

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="ML Classification / Regression",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #008B8B;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #008B8B;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #008B8B;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #008B8B;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .algorithm-card {
        background-color: #008B8B;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .user-info {
        background-color: #008B8B;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .dataframe-table {
        font-size: 0.9rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-message.user {
        background-color: #008B8B;
        border-left: 4px solid #007bff;
    }
    .chat-message.assistant {
        background-color: #008B8B;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Initialize Authentication --------------------
init_authentication()

# -------------------- Session State Initialization --------------------
def init_session_state():
    """Initialize all required session state variables."""
    defaults = {
        'model_trained': False,
        'model': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'label_encoders': {},
        'scaler': None,
        'imputer': None,
        'problem_type': None,
        'target_col': None,
        'feature_names': [],
        'df': None,
        'fitted_imputer': None,
        'chat_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------- Enhanced Data Display Functions --------------------
def style_dtype_info(row):
    """Style function for dtype info dataframe."""
    styles = []
    for i, value in enumerate(row):
        if row.name == 'Null Percentage' and value > 0:
            styles.append('background-color: #fff3cd')
        elif row.name == 'Null Count' and value > 0:
            styles.append('background-color: #f8d7da')
        else:
            styles.append('')
    return styles

def display_data_overview(df):
    """Enhanced data overview display."""
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("No data loaded. Please upload a dataset first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Dataset preview with tabs
    st.subheader("Dataset Preview")
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample"])
    
    with preview_tab1:
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    with preview_tab2:
        st.dataframe(df.tail(10), use_container_width=True, height=400)
    
    with preview_tab3:
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True, height=400)
    
    # Data types information with enhanced display
    st.subheader("Data Types Summary")
    dtype_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    # Apply styling correctly
    def highlight_missing(val):
        """Highlight missing values in the dataframe."""
        if isinstance(val, (int, float)) and val > 0:
            return 'background-color: #fff3cd'
        return ''
    
    # Style the dataframe properly
    styled_dtype_info = dtype_info.style.applymap(
        highlight_missing, 
        subset=['Null Count', 'Null Percentage']
    )
    
    st.dataframe(styled_dtype_info, use_container_width=True, height=400)
    
    # Basic statistics with tabs
    st.subheader("Statistical Summary")
    stat_tab1, stat_tab2 = st.tabs(["Numerical Columns", "Categorical Columns"])
    
    with stat_tab1:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe(), use_container_width=True, height=400)
        else:
            st.info("No numerical columns found in the dataset")
    
    with stat_tab2:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_stats = []
            for col in categorical_cols:
                cat_stats.append({
                    'Column': col,
                    'Unique Count': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].empty else 'N/A',
                    'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                    'Missing Values': df[col].isnull().sum()
                })
            cat_stats_df = pd.DataFrame(cat_stats)
            st.dataframe(cat_stats_df, use_container_width=True, height=400)
        else:
            st.info("No categorical columns found in the dataset")

def display_enhanced_eda(df):
    """Enhanced EDA with more visualizations."""
    st.markdown('<h2 class="section-header">📈 Enhanced Data Analysis</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("No data loaded. Please upload a dataset first.")
        return
    
    # Correlation heatmap with options
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=ax, mask=mask, fmt='.2f')
            ax.set_title('Correlation Heatmap (Upper Triangle)')
            st.pyplot(fig)
        
        with col2:
            st.write("**Top Correlations:**")
            # Get top correlations
            corr_pairs = correlation_matrix.unstack()
            sorted_pairs = corr_pairs.sort_values(key=abs, ascending=False)
            # Remove diagonal and duplicates
            sorted_pairs = sorted_pairs[sorted_pairs != 1.0]
            top_corrs = sorted_pairs.head(10)
            
            for (col1, col2), value in top_corrs.items():
                st.write(f"• {col1} - {col2}: {value:.3f}")
    else:
        st.warning("Not enough numeric columns for correlation analysis")
    
    # Distribution analysis with more options
    st.subheader("Distribution Analysis")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        selected_col = st.selectbox("Select column for distribution", numeric_cols)
    
    with dist_col2:
        plot_type = st.selectbox("Plot Type", ["Histogram + Box Plot", "Violin Plot", "Density Plot"])
    
    if selected_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Histogram + Box Plot":
            # Create subplots for histogram and boxplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            df[selected_col].hist(bins=30, ax=ax1, alpha=0.7, edgecolor='black')
            df[selected_col].plot.kde(ax=ax1, secondary_y=True, color='red')
            ax1.set_title(f'Distribution of {selected_col}')
            ax1.set_xlabel(selected_col)
            ax1.set_ylabel('Frequency')
            
            # Box plot
            df.boxplot(column=selected_col, ax=ax2)
            ax2.set_title(f'Box Plot of {selected_col}')
            
        elif plot_type == "Violin Plot":
            sns.violinplot(y=df[selected_col], ax=ax)
            ax.set_title(f'Violin Plot of {selected_col}')
            
        else:  # Density Plot
            df[selected_col].plot.kde(ax=ax)
            ax.set_title(f'Density Plot of {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Density')
        
        st.pyplot(fig)
    
    # Enhanced categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Variable Analysis")
        
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            cat_col = st.selectbox("Select categorical column", categorical_cols)
        
        with cat_col2:
            max_categories = st.slider("Max categories to display", 5, 20, 10)
        
        if cat_col:
            value_counts = df[cat_col].value_counts().head(max_categories)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar chart
            value_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title(f'Top {len(value_counts)} Categories in {cat_col}')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylabel('Count')
            
            # Pie chart (only if not too many categories)
            if len(value_counts) <= 10:
                value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, startangle=90)
                ax2.set_title(f'Distribution of {cat_col}')
                ax2.set_ylabel('')
            else:
                ax2.text(0.5, 0.5, 'Too many categories\nfor pie chart', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Distribution Display')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show value counts table
            st.write("**Value Counts:**")
            st.dataframe(value_counts, use_container_width=True)

# -------------------- Main App --------------------
class AutoMLApp:
    def __init__(self):
        # Use session state for persistence
        self.df = st.session_state.df
        self.model = st.session_state.model
        self.X_train = st.session_state.X_train
        self.X_test = st.session_state.X_test
        self.y_train = st.session_state.y_train
        self.y_test = st.session_state.y_test
        self.label_encoders = st.session_state.label_encoders
        self.scaler = st.session_state.scaler
        self.imputer = st.session_state.imputer
        self.fitted_imputer = st.session_state.fitted_imputer
        self.problem_type = st.session_state.problem_type
        self.target_col = st.session_state.target_col
        self.feature_names = st.session_state.feature_names
        
    def update_session_state(self):
        """Update session state with current instance values"""
        st.session_state.df = self.df
        st.session_state.model = self.model
        st.session_state.X_train = self.X_train
        st.session_state.X_test = self.X_test
        st.session_state.y_train = self.y_train
        st.session_state.y_test = self.y_test
        st.session_state.label_encoders = self.label_encoders
        st.session_state.scaler = self.scaler
        st.session_state.imputer = self.imputer
        st.session_state.fitted_imputer = self.fitted_imputer
        st.session_state.problem_type = self.problem_type
        st.session_state.target_col = self.target_col
        st.session_state.feature_names = self.feature_names
        st.session_state.model_trained = self.model is not None

    def run(self):
        # Show login form if not authenticated
        if not st.session_state.authenticated:
            login_form()
            return
        
        # Display user info and logout button
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f'<div class="user-info">👤 Logged in as: <strong>{st.session_state.current_user}</strong></div>', 
                       unsafe_allow_html=True)
        with col2:
            if st.button("🔄 Switch User"):
                logout()
                st.rerun()
        with col3:
            if st.button("🚪 Logout"):
                logout()
                st.rerun()
                return
        
        st.markdown('<h1 class="main-header">🤖 ML Classification / Regression</h1>', unsafe_allow_html=True)
        
        # Navigation
        nav_options = ["📊 Data Upload & Overview", "🔧 Data Preprocessing", "📈 EDA & Visualization", 
                      "🤖 Model Training", "📊 Model Evaluation", "🎯 Prediction", "💾 Save/Load Model", "🤖 AI Assistant"]
        selected_nav = st.sidebar.radio("Navigation", nav_options)
        
        # File upload in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset (CSV)", 
            type=['csv'],
            help="Upload a CSV file for analysis"
        )
        
        if uploaded_file is not None:
            try:
                self.df = pd.read_csv(uploaded_file)
                st.session_state.df = self.df
                st.sidebar.success(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                return
        elif self.df is None:
            st.info("👆 Please upload a CSV file to get started")
            return
        
        # Model training status indicator
        if st.session_state.model_trained:
            st.sidebar.success("✅ Model Trained")
            st.sidebar.write(f"**Problem Type:** {st.session_state.problem_type}")
            st.sidebar.write(f"**Target:** {st.session_state.target_col}")
        else:
            st.sidebar.warning("⏳ Model Not Trained")
        
        # User management (admin only)
        user_management_section()
        
        # Navigation routing
        if selected_nav == "📊 Data Upload & Overview":
            self.data_overview()
        elif selected_nav == "🔧 Data Preprocessing":
            self.data_preprocessing()
        elif selected_nav == "📈 EDA & Visualization":
            self.eda_visualization()
        elif selected_nav == "🤖 Model Training":
            self.model_training()
        elif selected_nav == "📊 Model Evaluation":
            self.model_evaluation()
        elif selected_nav == "🎯 Prediction":
            self.prediction_interface()
        elif selected_nav == "💾 Save/Load Model":
            self.save_load_model()
        elif selected_nav == "🤖 AI Assistant":
            ai_assistant_chat(self)
        
        self.update_session_state()
    
    def data_overview(self):
        """Enhanced data overview with better display"""
        display_data_overview(self.df)
    
    def data_preprocessing(self):
        st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
        
        if self.df is None:
            st.error("No data loaded. Please upload a dataset first.")
            return
        
        # Missing values treatment
        st.subheader("Missing Values Treatment")
        if self.df.isnull().sum().sum() > 0:
            missing_cols = self.df.columns[self.df.isnull().any()].tolist()
            st.warning(f"Missing values found in: {', '.join(missing_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                num_strategy = st.selectbox("Numerical columns strategy", 
                                          ["mean", "median", "most_frequent", "drop"])
            with col2:
                cat_strategy = st.selectbox("Categorical columns strategy", 
                                          ["most_frequent", "drop"])
            
            if st.button("Apply Missing Value Treatment"):
                df_copy = self.df.copy()
                for col in missing_cols:
                    if df_copy[col].dtype in ['int64', 'float64']:
                        if num_strategy == 'drop':
                            df_copy = df_copy.dropna(subset=[col])
                        else:
                            imputer = SimpleImputer(strategy=num_strategy)
                            df_copy[col] = imputer.fit_transform(df_copy[[col]]).ravel()
                    else:
                        if cat_strategy == 'drop':
                            df_copy = df_copy.dropna(subset=[col])
                        else:
                            imputer = SimpleImputer(strategy=cat_strategy)
                            df_copy[col] = imputer.fit_transform(df_copy[[col]]).ravel()
                self.df = df_copy
                st.session_state.df = self.df
                st.success("✅ Missing values treated successfully!")
        else:
            st.success("✅ No missing values found!")
        
        # Remove duplicates
        st.subheader("Duplicate Handling")
        duplicates = self.df.duplicated().sum()
        st.write(f"Duplicate rows found: {duplicates}")
        if duplicates > 0 and st.button("Remove Duplicates"):
            self.df = self.df.drop_duplicates()
            st.session_state.df = self.df
            st.success(f"✅ Removed {duplicates} duplicate rows!")
        
        # Column type management
        st.subheader("Column Type Management")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Column Types:**")
            for col in self.df.columns:
                st.write(f"- {col}: {self.df[col].dtype}")
        
        with col2:
            columns_to_drop = st.multiselect("Select columns to drop", self.df.columns)
            if st.button("Drop Selected Columns") and columns_to_drop:
                self.df = self.df.drop(columns=columns_to_drop)
                st.session_state.df = self.df
                st.success(f"✅ Dropped columns: {', '.join(columns_to_drop)}")
        
        # Feature engineering
        st.subheader("Feature Engineering")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Create Date Features (if datetime columns exist)"):
                df_copy = self.df.copy()
                # Convert object columns that look like dates
                for col in df_copy.select_dtypes(include=['object']).columns:
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col])
                        df_copy[f'{col}_year'] = df_copy[col].dt.year
                        df_copy[f'{col}_month'] = df_copy[col].dt.month
                        df_copy[f'{col}_day'] = df_copy[col].dt.day
                        st.success(f"✅ Date features created from {col}!")
                    except:
                        continue
                self.df = df_copy
                st.session_state.df = self.df
        
        with col2:
            if st.button("Create Age Groups (if 'Age' column exists)"):
                if 'Age' in self.df.columns:
                    df_copy = self.df.copy()
                    df_copy['AgeGroup'] = pd.cut(df_copy['Age'], 
                                               bins=[0, 18, 35, 50, 100], 
                                               labels=['0-18', '19-35', '36-50', '51+'])
                    self.df = df_copy
                    st.session_state.df = self.df
                    st.success("✅ Age groups created!")
                else:
                    st.warning("No 'Age' column found")
        
        st.success("✅ Data preprocessing completed!")
        st.dataframe(self.df.head(), use_container_width=True)
    
    def eda_visualization(self):
        """Enhanced EDA visualization"""
        display_enhanced_eda(self.df)
    
    def can_use_stratification(self, y):
        """Check if stratification can be used for the target variable"""
        if len(np.unique(y)) < 2:
            return False
        
        # Check if all classes have at least 2 samples
        value_counts = pd.Series(y).value_counts()
        return all(value_counts >= 2)
    
    def detect_problem_type(self, y):
        """Automatically detect if problem is classification or regression"""
        unique_values = len(np.unique(y))
        
        if unique_values <= 10:  # Classification for small number of unique values
            return "Classification"
        else:  # Regression for many unique values
            return "Regression"
    
    def get_algorithm_description(self, algorithm_name, problem_type):
        """Get description for each algorithm"""
        descriptions = {
            "Classification": {
                "K-Nearest Neighbors": "Instance-based learning that classifies based on nearest neighbors",
                "Naive Bayes": "Probabilistic classifier based on Bayes' theorem with independence assumptions",
                "Decision Tree": "Tree-like model that makes decisions based on feature thresholds",
                "Random Forest": "Ensemble of decision trees for improved accuracy and robustness",
                "Logistic Regression": "Linear model for binary and multiclass classification",
                "SVM": "Finds optimal hyperplane to separate classes with maximum margin"
            },
            "Regression": {
                "K-Nearest Neighbors": "Predicts based on average of k nearest neighbors' values",
                "Decision Tree": "Tree-like model that predicts continuous values",
                "Random Forest": "Ensemble of decision trees for regression tasks",
                "Linear Regression": "Linear approach to modeling relationship between variables",
                "SVR": "Support Vector Machine for regression tasks"
            }
        }
        
        return descriptions.get(problem_type, {}).get(algorithm_name, "No description available")
    
    def model_training(self):
        st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if self.df is None:
            st.error("No data loaded. Please upload a dataset first.")
            return
        
        # Problem type selection
        st.subheader("Problem Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            # Auto-detect problem type first
            target_col = st.selectbox("Select Target Variable", self.df.columns)
            if target_col:
                y_sample = self.df[target_col].dropna()
                auto_detected = self.detect_problem_type(y_sample)
            else:
                auto_detected = "Classification"
            
            problem_type = st.radio("Select Problem Type", 
                                  ["Classification", "Regression"],
                                  index=0 if auto_detected == "Classification" else 1)
            self.problem_type = problem_type
            self.target_col = target_col
        
        with col2:
            if target_col:
                st.info(f"**Target Variable Info:** {self.df[target_col].nunique()} unique values, "
                       f"{self.df[target_col].isnull().sum()} missing values")
                
                # Show value distribution for classification
                if problem_type == "Classification":
                    value_counts = self.df[target_col].value_counts()
                    st.write("**Class Distribution:**")
                    st.dataframe(value_counts)
                    
                    # Check if stratification is possible
                    can_stratify = self.can_use_stratification(self.df[target_col].dropna())
                    if not can_stratify:
                        st.error("⚠️ Cannot use stratification: Some classes have less than 2 samples")
        
        # Feature selection
        st.subheader("Feature Selection")
        available_features = [col for col in self.df.columns if col != target_col]
        selected_features = st.multiselect("Select Features for Model", 
                                         available_features, 
                                         default=available_features)
        
        if not selected_features:
            st.error("Please select at least one feature!")
            return
        
        self.feature_names = selected_features
        
        # Model selection
        st.subheader("Model Selection")
        
        if problem_type == "Classification":
            algorithms = [
                "K-Nearest Neighbors", 
                "Naive Bayes", 
                "Decision Tree", 
                "Random Forest", 
                "Logistic Regression", 
                "SVM"
            ]
        else:
            algorithms = [
                "K-Nearest Neighbors", 
                "Decision Tree", 
                "Random Forest", 
                "Linear Regression", 
                "SVR"
            ]
        
        model_choice = st.selectbox("Select Algorithm", algorithms)
        
        # Show algorithm description
        description = self.get_algorithm_description(model_choice, problem_type)
        st.markdown(f'<div class="algorithm-card"><strong>{model_choice}:</strong> {description}</div>', 
                   unsafe_allow_html=True)
        
        # Hyperparameters
        st.subheader("Hyperparameter Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
            random_state = st.number_input("Random State", value=42)
        
        with col2:
            if model_choice in ["Random Forest", "Decision Tree"]:
                max_depth = st.slider("Max Depth", 3, 20, 10)
                if model_choice == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 10, 200, 100)
            
            elif model_choice == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of Neighbors", 1, 15, 5)
            
            elif model_choice == "SVM" or model_choice == "SVR":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
                C = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
        
        # Prepare data
        X = self.df[selected_features].copy()
        y = self.df[target_col].copy()
        
        # Remove rows where target is missing
        missing_target = y.isnull()
        if missing_target.any():
            st.warning(f"Removing {missing_target.sum()} rows with missing target values")
            X = X[~missing_target]
            y = y[~missing_target]
        
        # Check if we have enough data
        if len(y) < 10:
            st.error("❌ Not enough data after cleaning. Need at least 10 samples.")
            return
        
        # Preprocessing
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values in features
        if X.isnull().any().any():
            self.imputer = SimpleImputer(strategy='median')
            X_imputed = self.imputer.fit_transform(X)
            self.fitted_imputer = self.imputer
        else:
            self.imputer = SimpleImputer(strategy='median')
            self.fitted_imputer = self.imputer.fit(X)
            X_imputed = X.values
        
        # Scale features
        self.scaler = StandardScaler()
        if len(X_imputed) > 0:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            st.error("❌ No data available after preprocessing")
            return
        
        # Determine if stratification can be used
        use_stratify = False
        if problem_type == "Classification":
            use_stratify = self.can_use_stratification(y)
        
        try:
            # Split data with error handling
            if use_stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X_scaled, y, test_size=test_size/100, random_state=random_state,
                    stratify=y
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X_scaled, y, test_size=test_size/100, random_state=random_state
                )
        except ValueError as e:
            st.error(f"❌ Error during train-test split: {str(e)}")
            st.info("Try increasing test size or selecting a different target variable")
            return
        
        # Model initialization
        if problem_type == "Classification":
            if model_choice == "K-Nearest Neighbors":
                self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_choice == "Naive Bayes":
                self.model = GaussianNB()
            elif model_choice == "Decision Tree":
                self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            elif model_choice == "Random Forest":
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    random_state=random_state
                )
            elif model_choice == "Logistic Regression":
                self.model = LogisticRegression(random_state=random_state)
            else:  # SVM
                self.model = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
        else:  # Regression
            if model_choice == "K-Nearest Neighbors":
                self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
            elif model_choice == "Decision Tree":
                self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
            elif model_choice == "Random Forest":
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
            elif model_choice == "Linear Regression":
                self.model = LinearRegression()
            else:  # SVR
                self.model = SVR(kernel=kernel, C=C)
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                try:
                    self.model.fit(self.X_train, self.y_train)
                    
                    # Cross-validation with appropriate scoring
                    cv_folds = min(5, len(np.unique(y)))  # Adjust folds for small datasets
                    if problem_type == 'Classification':
                        scoring = 'accuracy'
                    else:
                        scoring = 'r2'
                    
                    if len(y) >= 5:  # Only do CV if we have enough data
                        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring=scoring)
                        cv_display = f"{cv_scores.mean():.3f} (±{cv_scores.std():.3f})"
                    else:
                        cv_display = "Not enough data for CV"
                    
                    st.success("✅ Model trained successfully!")
                    st.session_state.model_trained = True
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training Samples", len(self.X_train))
                    with col2:
                        st.metric("Test Samples", len(self.X_test))
                    with col3:
                        st.metric("CV Score", cv_display)
                        
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
    
    def model_evaluation(self):
        st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained or self.model is None:
            st.error("❌ Please train a model first in the 'Model Training' section")
            st.info("Go to the '🤖 Model Training' tab to train your model")
            return
        
        if self.X_test is None or self.y_test is None:
            st.error("No test data available for evaluation")
            return
        
        # Show model info
        st.subheader("Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Problem Type", self.problem_type)
        with col2:
            st.metric("Target Variable", self.target_col)
        with col3:
            st.metric("Features Used", len(self.feature_names))
        
        # Predictions
        try:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test) if hasattr(self.model, "predict_proba") else None
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return
        
        if self.problem_type == "Classification":
            # Classification metrics
            st.subheader("Classification Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = accuracy_score(self.y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col2:
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                st.metric("Precision", f"{precision:.3f}")
            
            with col3:
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                st.metric("Recall", f"{recall:.3f}")
            
            with col4:
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(self.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
        else:
            # Regression metrics
            st.subheader("Regression Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mse = mean_squared_error(self.y_test, y_pred)
                st.metric("MSE", f"{mse:.3f}")
            
            with col2:
                rmse = np.sqrt(mse)
                st.metric("RMSE", f"{rmse:.3f}")
            
            with col3:
                r2 = r2_score(self.y_test, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
            
            with col4:
                mae = mean_absolute_error(self.y_test, y_pred)
                st.metric("MAE", f"{mae:.3f}")
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.y_test, y_pred, alpha=0.5)
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
        
        # Feature Importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            st.subheader("Feature Importance")
            n_features = len(self.model.feature_importances_)
            feature_names_to_use = self.feature_names[:n_features]
            
            if len(feature_names_to_use) < n_features:
                feature_names_to_use = [f'Feature {i}' for i in range(n_features)]
            
            feature_importance = pd.DataFrame({
                'feature': feature_names_to_use,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Most Important Features')
            st.pyplot(fig)
    
    def prediction_interface(self):
        st.markdown('<h2 class="section-header">🎯 Make Predictions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained or self.model is None:
            st.error("❌ Please train a model first in the 'Model Training' section")
            st.info("Go to the '🤖 Model Training' tab to train your model")
            return
        
        st.subheader("Model Prediction Interface")
        st.success(f"Model ready for predictions! (Trained on {self.problem_type} problem)")
        
        # Create input form based on feature names
        st.subheader("Input Features")
        input_data = {}
        
        if not self.feature_names:
            st.error("No feature information available. Please retrain the model.")
            return
        
        # Create input fields for each feature
        for i, feature in enumerate(self.feature_names):
            if feature in self.df.columns:
                if self.df[feature].dtype in ['int64', 'float64']:
                    min_val = float(self.df[feature].min())
                    max_val = float(self.df[feature].max())
                    default_val = float(self.df[feature].median())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        key=f"input_{feature}"
                    )
                else:
                    unique_vals = self.df[feature].unique()
                    selected_val = st.selectbox(f"{feature}", unique_vals, key=f"select_{feature}")
                    input_data[feature] = selected_val
        
        if st.button("🔮 Predict", key="predict_button"):
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Preprocess input data same as training
                for col in input_df.columns:
                    if col in self.label_encoders:
                        try:
                            input_df[col] = self.label_encoders[col].transform(input_df[col])
                        except ValueError:
                            most_common = self.df[col].mode()[0]
                            input_df[col] = self.label_encoders[col].transform([most_common])[0]
                
                # Ensure correct feature order
                processed_input = []
                for feature in self.feature_names:
                    if feature in input_df.columns:
                        processed_input.append(input_df[feature].values[0])
                    else:
                        if self.df[feature].dtype in ['int64', 'float64']:
                            processed_input.append(self.df[feature].median())
                        else:
                            processed_input.append(self.df[feature].mode()[0])
                
                # Transform input using fitted transformers
                if self.fitted_imputer is not None:
                    input_imputed = self.fitted_imputer.transform([processed_input])
                else:
                    input_imputed = [processed_input]
                
                # Scale the input
                input_scaled = self.scaler.transform(input_imputed)
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                
                # Display results
                st.success("### Prediction Results")
                
                if self.problem_type == "Classification":
                    if hasattr(self.model, "predict_proba"):
                        probability = self.model.predict_proba(input_scaled)[0]
                        max_prob = max(probability)
                        predicted_class = prediction
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Class", predicted_class)
                        with col2:
                            st.metric("Confidence", f"{max_prob:.3f}")
                        
                        st.subheader("Class Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': self.model.classes_,
                            'Probability': probability
                        }).sort_values('Probability', ascending=False)
                        st.dataframe(prob_df)
                    else:
                        st.metric("Predicted Class", prediction)
                else:
                    st.metric("Predicted Value", f"{prediction:.3f}")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("This might be due to feature mismatch. Try retraining the model.")
    
    def save_load_model(self):
        st.markdown('<h2 class="section-header">💾 Model Management</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Save Model")
            if st.session_state.model_trained and self.model is not None:
                model_filename = st.text_input("Model filename", "automl_model.pkl")
                
                if st.button("💾 Save Model"):
                    try:
                        # Create model artifact
                        model_artifact = {
                            'model': self.model,
                            'label_encoders': self.label_encoders,
                            'fitted_imputer': self.fitted_imputer,
                            'scaler': self.scaler,
                            'problem_type': self.problem_type,
                            'target_col': self.target_col,
                            'feature_names': self.feature_names,
                            'df_columns': self.df.columns.tolist() if self.df is not None else [],
                            'timestamp': datetime.now()
                        }
                        
                        joblib.dump(model_artifact, model_filename)
                        st.success(f"✅ Model saved as {model_filename}")
                        
                        with open(model_filename, "rb") as file:
                            btn = st.download_button(
                                label="📥 Download Model",
                                data=file,
                                file_name=model_filename,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
            else:
                st.warning("No trained model to save")
        
        with col2:
            st.subheader("Load Model")
            uploaded_model = st.file_uploader("Upload trained model", type=['pkl'], key="model_uploader")
            
            if uploaded_model is not None:
                if st.button("📥 Load Model"):
                    try:
                        model_artifact = joblib.load(uploaded_model)
                        
                        # Update all session state variables
                        self.model = model_artifact['model']
                        self.label_encoders = model_artifact['label_encoders']
                        self.fitted_imputer = model_artifact.get('fitted_imputer', None)
                        self.scaler = model_artifact['scaler']
                        self.problem_type = model_artifact.get('problem_type', 'Classification')
                        self.target_col = model_artifact.get('target_col', None)
                        self.feature_names = model_artifact.get('feature_names', [])
                        
                        # Update session state
                        self.update_session_state()
                        st.session_state.model_trained = True
                        
                        st.success("✅ Model loaded successfully!")
                        st.info(f"Loaded {self.problem_type} model for target '{self.target_col}'")
                        
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")

# -------------------- Run the App --------------------
if __name__ == "__main__":
    # Initialize session state
    init_session_state()
    
    # Create and run the app
    app = AutoMLApp()
    app.run()