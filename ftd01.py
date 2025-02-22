import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# ---------------------------
# Session State Initialization
# ---------------------------
if "users" not in st.session_state:
    st.session_state["users"] = {}
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "otp" not in st.session_state:
    st.session_state["otp"] = None
if "temp_user" not in st.session_state:
    st.session_state["temp_user"] = {}
if "show_otp_input" not in st.session_state:
    st.session_state["show_otp_input"] = False
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# ---------------------------
# Helper Functions
# ---------------------------
def generate_and_display_otp():
    """Generate OTP and display it (simulating sending)."""
    otp = random.randint(100000, 999999)
    st.session_state["otp"] = otp
    st.info(f"🔐 Your OTP is: {otp} (In a real app, this would be sent to your phone)")
    return otp

def preprocess_data(df):
    """Preprocess the dataset for fake profile detection."""
    drop_columns = ["id", "name", "screen_name", "created_at", "updated", "profile_image_url", 
                    "profile_banner_url", "profile_background_image_url_https", "profile_image_url_https", 
                    "profile_background_image_url", "description"]
    df_cleaned = df.drop(columns=drop_columns, errors='ignore')
    df_cleaned["location"].fillna("Unknown", inplace=True)
    df_cleaned["default_profile"].fillna(0, inplace=True)
    df_cleaned["profile_background_tile"].fillna(0, inplace=True)
    df_cleaned["location"] = df_cleaned["location"].astype("category").cat.codes
    return df_cleaned

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate the machine learning models."""
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb)

    return rf_clf, xgb_clf, accuracy_rf, report_rf, accuracy_xgb, report_xgb

def create_prediction_visualizations(input_data, probability, prediction_history):
    """Create visualizations for the prediction results."""
    # Pie chart for current prediction probability
    fig_pie = px.pie(
        values=[probability[1], probability[0]],
        names=['Fake', 'Real'],
        title='Prediction Probability Distribution',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    # Line chart for feature comparison
    feature_names = ['Statuses', 'Followers', 'Friends', 'Location', 'Default Profile', 'Background Tile']
    fig_features = go.Figure()
    fig_features.add_trace(go.Scatter(
        x=feature_names,
        y=input_data[0],
        mode='lines+markers',
        name='Current Profile',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10)
    ))
    fig_features.update_layout(
        title='Feature Distribution',
        xaxis_title='Features',
        yaxis_title='Value',
        hovermode='x'
    )
    
    # Historical predictions line chart
    if prediction_history:
        history_df = pd.DataFrame(prediction_history)
        fig_history = px.line(
            history_df,
            x='timestamp',
            y='confidence',
            color='prediction',
            title='Prediction History',
            labels={'confidence': 'Confidence Score', 'prediction': 'Prediction Type'},
            color_discrete_map={'Fake Profile': '#FF6B6B', 'Real Profile': '#4ECDC4'}
        )
    else:
        fig_history = None
    
    return fig_pie, fig_features, fig_history

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Fake Profile Detection", layout="wide")

# ---------------------------
# Navigation
# ---------------------------
page = st.sidebar.selectbox("Navigation", ["Login", "Register", "Fake Profile Detection"])

# ---------------------------
# Registration Page
# ---------------------------
if page == "Register":
    st.title("User Registration")
    with st.container():
        username = st.text_input("Username", key="reg_username")
        phone_number = st.text_input("Phone Number (with country code)", key="reg_phone")
        password = st.text_input("Password", type="password", key="reg_password")
        
        if st.button("Register"):
            if username in st.session_state["users"]:
                st.error("Username already exists!")
            elif not username or not phone_number or not password:
                st.error("Please fill in all fields.")
            else:
                generate_and_display_otp()
                st.session_state["temp_user"] = {
                    "username": username,
                    "phone_number": phone_number,
                    "password": password
                }
                st.session_state["show_otp_input"] = True

        if st.session_state["show_otp_input"]:
            otp_input = st.text_input("Enter OTP")
            if st.button("Verify OTP"):
                if st.session_state.get("otp") and otp_input:
                    try:
                        if int(otp_input) == st.session_state["otp"]:
                            user_details = st.session_state["temp_user"]
                            st.session_state["users"][user_details["username"]] = user_details
                            st.success("Registration successful! Please proceed to login.")
                            st.session_state["otp"] = None
                            st.session_state["show_otp_input"] = False
                            st.session_state["temp_user"] = {}
                        else:
                            st.error("Incorrect OTP. Please try again.")
                    except ValueError:
                        st.error("OTP should be numeric.")

# ---------------------------
# Login Page
# ---------------------------
if page == "Login":
    st.title("User Login")
    with st.container():
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            users = st.session_state["users"]
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = username
                st.success(f"Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

# ---------------------------
# Fake Profile Detection Page (Protected)
# ---------------------------
if page == "Fake Profile Detection":
    if not st.session_state["logged_in"]:
        st.error("Please log in to access this page.")
    else:
        st.title("Fake Profile Detection App")
        
        # Add logout button in the sidebar
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["current_user"] = None
            st.experimental_rerun()
        
        # Add clear history button
        if st.sidebar.button("Clear Prediction History"):
            st.session_state["prediction_history"] = []
            st.success("Prediction history cleared!")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            df_cleaned = preprocess_data(df)
            features = ["statuses_count", "followers_count", "friends_count", "location", 
                       "default_profile", "profile_background_tile"]
            
            X = df_cleaned[features]
            y = df_cleaned["Label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    rf_clf, xgb_clf, accuracy_rf, report_rf, accuracy_xgb, report_xgb = train_models(
                        X_train, X_test, y_train, y_test
                    )
                    st.session_state["rf_clf"] = rf_clf
                    st.session_state["xgb_clf"] = xgb_clf
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Random Forest Results:")
                        st.write(f"Accuracy: {accuracy_rf:.2f}")
                        st.text(report_rf)
                    
                    with col2:
                        st.write("### XGBoost Results:")
                        st.write(f"Accuracy: {accuracy_xgb:.2f}")
                        st.text(report_xgb)
        
        st.write("## Predict Fake Profile")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                statuses_count = st.number_input("Statuses Count", min_value=0)
                followers_count = st.number_input("Followers Count", min_value=0)
                friends_count = st.number_input("Friends Count", min_value=0)
            
            with col2:
                location = st.number_input("Location Code", min_value=0)
                default_profile = st.selectbox("Default Profile", [0, 1])
                profile_background_tile = st.selectbox("Profile Background Tile", [0, 1])
            
            submitted = st.form_submit_button("Predict")
            if submitted:
                if "rf_clf" not in st.session_state:
                    st.error("Please train the model first!")
                else:
                    input_data = np.array([[
                        statuses_count, followers_count, friends_count,
                        location, default_profile, profile_background_tile
                    ]])
                    prediction = st.session_state["rf_clf"].predict(input_data)
                    probability = st.session_state["rf_clf"].predict_proba(input_data)[0]
                    
                    result = "Fake Profile" if prediction[0] == 1 else "Real Profile"
                    confidence = probability[1] if prediction[0] == 1 else probability[0]
                    
                    # Add prediction to history
                    st.session_state["prediction_history"].append({
                        'timestamp': pd.Timestamp.now(),
                        'prediction': result,
                        'confidence': confidence,
                        'features': input_data[0].tolist()
                    })
                    
                    # Create visualizations
                    fig_pie, fig_features, fig_history = create_prediction_visualizations(
                        input_data, probability, st.session_state["prediction_history"]
                    )
                    
                    st.write("### Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", result)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Display visualizations
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.plotly_chart(fig_features, use_container_width=True)
                    
                    if fig_history:
                        st.plotly_chart(fig_history, use_container_width=True)
        
        # Display prediction history table
        if st.session_state["prediction_history"]:
            st.write("### Prediction History")
            history_df = pd.DataFrame(st.session_state["prediction_history"])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                history_df[['timestamp', 'prediction', 'confidence']],
                use_container_width=True
            )
