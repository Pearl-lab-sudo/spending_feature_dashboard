import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Load .env variables (for local development)
load_dotenv()

# Get database credentials from environment variables or Streamlit secrets
try:
    # Try Streamlit secrets first (for deployment)
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
except:
    # Fallback to environment variables (for local development)
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

# Validate that all required credentials are available
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    st.error("❌ Database credentials are missing. Please check your secrets configuration.")
    st.stop()

# Create connection string
conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(conn_str)
except Exception as e:
    st.error(f"❌ Failed to create database connection: {str(e)}")
    st.stop()

# --- CONFIGURATION ---
LADDER_BLUE = "#0039A6"
LADDER_WHITE = "#FFFFFF"
st.set_page_config(page_title="New Users Spending Feature Dashboard", layout="wide", page_icon="💸")

# --- STYLING ---
st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {LADDER_WHITE};
            color: black;
        }}
        .sidebar .sidebar-content {{
            background-color: {LADDER_BLUE};
            color: white;
        }}
        .css-1aumxhk {{
            color: {LADDER_BLUE} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Ladder Spending Feature - New Users Analysis")
st.markdown("""
This dashboard highlights new users from **June 1, 2025**, onward and their interactions with the **Spending Feature** (Budgets & Transactions).
""")

# --- GET DATE RANGE FOR FILTERS ---
@st.cache_data
def get_date_range():
    try:
        query = "SELECT MIN(created_at) as min_date, MAX(created_at) as max_date FROM users"
        date_range_df = pd.read_sql(query, engine)
        min_date = pd.to_datetime(date_range_df['min_date'].iloc[0]).date()
        max_date = datetime.today().date()  # Use today's date as max
        return min_date, max_date
    except Exception as e:
        st.error(f"❌ Failed to get date range: {str(e)}")
        # Return default date range if query fails
        return datetime(2020, 1, 1).date(), datetime.today().date()

# --- GET TOTAL USER COUNT ---
@st.cache_data
def get_total_user_count():
    try:
        query = "SELECT COUNT(*) as total_users FROM users"
        total_users_df = pd.read_sql(query, engine)
        return total_users_df['total_users'].iloc[0]
    except Exception as e:
        st.error(f"❌ Failed to get total user count: {str(e)}")
        return 0

# --- DATA LOADING ---
@st.cache_data
def load_data(start_date, end_date):
    try:
        query = f"""
        WITH new_users AS (
            SELECT id AS user_id, first_name, last_name, email, created_at 
            FROM users
            WHERE created_at >= '{start_date}' AND created_at <= '{end_date}'
        ),
        budget_acts AS (
            SELECT 
                b.user_id,
                MAX(b.created_at) AS last_budget_time
            FROM budgets b
            JOIN new_users u ON b.user_id = u.user_id
            GROUP BY b.user_id
        ),
        transaction_acts AS (
            SELECT 
                t.user_id,
                MAX(t.updated_at) AS last_transaction_time
            FROM manual_and_external_transactions t
            JOIN new_users u ON t.user_id = u.user_id
            GROUP BY t.user_id
        ),
        combined AS (
            SELECT 
                u.user_id,
                u.first_name || ' ' || u.last_name AS customer_name,
                u.email,
                u.created_at,
                COALESCE(b.last_budget_time, NULL) AS last_budget_time,
                COALESCE(t.last_transaction_time, NULL) AS last_transaction_time,
                CASE
                    WHEN b.last_budget_time IS NOT NULL AND t.last_transaction_time IS NOT NULL THEN 'Budget + Transaction'
                    WHEN b.last_budget_time IS NOT NULL THEN 'Budget'
                    WHEN t.last_transaction_time IS NOT NULL THEN 'Transaction'
                    ELSE 'None'
                END AS spending_feature_used,
                CASE
                    WHEN b.last_budget_time IS NOT NULL OR t.last_transaction_time IS NOT NULL THEN 'Yes'
                    ELSE 'No'
                END AS interacted_with_spending_feature
            FROM new_users u
            LEFT JOIN budget_acts b ON u.user_id = b.user_id
            LEFT JOIN transaction_acts t ON u.user_id = t.user_id
        )
        SELECT *
        FROM combined
        ORDER BY interacted_with_spending_feature DESC;
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Database query failed: {str(e)}")
        # Return empty dataframe with expected columns to prevent app crash
        return pd.DataFrame(columns=['user_id', 'customer_name', 'email', 'created_at', 
                                   'last_budget_time', 'last_transaction_time', 
                                   'spending_feature_used', 'interacted_with_spending_feature'])

# Get date range and total users
with st.spinner("Getting database information..."):
    min_date, max_date = get_date_range()
    total_users_overall = get_total_user_count()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📅 Filter Data")

# Date range filter with dynamic min/max dates
date_range = st.sidebar.date_input(
    "Signup Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help=f"Available date range: {min_date} to {max_date}"
)

# Handle date range input
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    # If only one date is selected, use it as both start and end
    start_date = end_date = date_range if hasattr(date_range, 'year') else min_date

# Load data with selected date range
with st.spinner("Loading filtered data from database..."):
    df = load_data(start_date, end_date)

if df.empty:
    st.warning("⚠️ No data available for the selected date range. Please try a different date range.")
    st.stop()

df['created_at'] = pd.to_datetime(df['created_at'])

# Interaction filter
interaction_filter = st.sidebar.selectbox("Interaction", ["All"] + df["interacted_with_spending_feature"].unique().tolist())

# Apply interaction filter
if interaction_filter != "All":
    df = df[df["interacted_with_spending_feature"] == interaction_filter]

st.sidebar.markdown("---")
st.sidebar.markdown("Built for Ladder ✨")

# --- KPI METRICS ---
total_new_users = df.shape[0]
interacted = df[df["interacted_with_spending_feature"] == "Yes"].shape[0]
no_interaction = total_new_users - interacted

# Display metrics in 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 Total Users (Overall)", 
        value=f"{total_users_overall:,}",
        help="Total number of users in the entire database"
    )

with col2:
    st.metric(
        label="🧑‍💼 Filtered Users", 
        value=f"{total_new_users:,}",
        help=f"Users created between {start_date} and {end_date}"
    )

with col3:
    st.metric(
        label="✅ Used Spending Feature", 
        value=f"{interacted:,}",
        delta=f"{(interacted/total_new_users*100):.1f}%" if total_new_users > 0 else "0%",
        help="Users who interacted with budgets or transactions"
    )

with col4:
    st.metric(
        label="❌ No Interaction", 
        value=f"{no_interaction:,}",
        delta=f"{(no_interaction/total_new_users*100):.1f}%" if total_new_users > 0 else "0%",
        help="Users who haven't used spending features yet"
    )

# --- TIME-SERIES CHART ---
st.subheader("📈 Time-Series: New Signups vs. Spending Interaction")

signup_timeseries = (
    df.groupby([df["created_at"].dt.date, "interacted_with_spending_feature"])
    .size()
    .reset_index(name="user_count")
    .rename(columns={"created_at": "signup_date"})
)

fig = go.Figure()

# Add trace for users who interacted
yes_data = signup_timeseries[signup_timeseries["interacted_with_spending_feature"] == "Yes"]
if not yes_data.empty:
    fig.add_trace(go.Scatter(
        x=yes_data["signup_date"],
        y=yes_data["user_count"],
        mode='lines+markers',
        name='Used Spending Feature',
        line=dict(color="#010414", width=3),
        marker=dict(size=6)
    ))

# Add trace for users who didn't interact
no_data = signup_timeseries[signup_timeseries["interacted_with_spending_feature"] == "No"]
if not no_data.empty:
    fig.add_trace(go.Scatter(
        x=no_data["signup_date"],
        y=no_data["user_count"],
        mode='lines+markers',
        name='No Interaction',
        line=dict(color="#FFA500", width=3),
        marker=dict(size=6)
    ))

fig.update_layout(
    title="Spending Feature Usage Over Time",
    xaxis_title="Signup Date",
    yaxis_title="User Count",
    plot_bgcolor="#1347AD",
    paper_bgcolor="#1347AD",
    font=dict(color='white'),
    hovermode='x unified',
    xaxis=dict(showgrid=False, color='white'),
    yaxis=dict(showgrid=False, color='white'),
    legend_title="Interaction Status",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- FEATURE USAGE CHART ---
st.subheader("💡 Spending Feature Usage Breakdown")
feature_counts = df["spending_feature_used"].value_counts().reset_index()
feature_counts.columns = ["Feature Type", "User Count"]

# Create a more detailed breakdown chart
fig_breakdown = px.pie(
    feature_counts, 
    values='User Count', 
    names='Feature Type',
    title='Distribution of Spending Feature Usage',
    color_discrete_sequence=['#0039A6', '#1347AD', '#FFA500', '#FF6B6B']
)

fig_breakdown.update_traces(textposition='inside', textinfo='percent+label')
fig_breakdown.update_layout(
    font=dict(size=12),
    title_font_size=16
)

st.plotly_chart(fig_breakdown, use_container_width=True)

# Also show the bar chart
st.bar_chart(feature_counts.set_index("Feature Type"))

# --- SUMMARY STATISTICS ---
st.subheader("📊 Summary Statistics")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Adoption Rate**: {(interacted/total_new_users*100):.1f}% of filtered users have used spending features
    
    **Most Popular Feature**: {feature_counts.iloc[0]['Feature Type']} ({feature_counts.iloc[0]['User Count']} users)
    
    **Date Range**: {start_date} to {end_date}
    """)

with col2:
    if total_new_users > 0:
        engagement_rate = (interacted / total_new_users) * 100
        if engagement_rate >= 50:
            st.success(f"🎉 Great engagement! {engagement_rate:.1f}% adoption rate")
        elif engagement_rate >= 25:
            st.warning(f"📈 Good progress! {engagement_rate:.1f}% adoption rate")
        else:
            st.error(f"📉 Low engagement: {engagement_rate:.1f}% adoption rate")

# --- TABLE VIEW ---
st.subheader("📋 Detailed User Activity Table")

# Add search functionality
search_term = st.text_input("🔍 Search users (name or email):", placeholder="Enter name or email to search...")

# Filter dataframe based on search
if search_term:
    mask = (
        df['customer_name'].str.contains(search_term, case=False, na=False) |
        df['email'].str.contains(search_term, case=False, na=False)
    )
    filtered_df = df[mask]
    st.info(f"Found {len(filtered_df)} users matching '{search_term}'")
else:
    filtered_df = df

# Display the table
st.dataframe(
    filtered_df,
    use_container_width=True,
    column_config={
        "created_at": st.column_config.DatetimeColumn(
            "Signup Date",
            format="DD/MM/YYYY HH:mm"
        ),
        "last_budget_time": st.column_config.DatetimeColumn(
            "Last Budget Activity",
            format="DD/MM/YYYY HH:mm"
        ),
        "last_transaction_time": st.column_config.DatetimeColumn(
            "Last Transaction Activity", 
            format="DD/MM/YYYY HH:mm"
        ),
        "interacted_with_spending_feature": st.column_config.SelectboxColumn(
            "Used Spending Feature",
            options=["Yes", "No"]
        )
    }
)