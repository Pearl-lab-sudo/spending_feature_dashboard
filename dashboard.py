import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import psycopg2
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
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    # Fallback to environment variables (for local development)
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
st.set_page_config(page_title="Ladder Analytics Dashboard", layout="wide", page_icon="📊")

# --- PAGE NAVIGATION ---
st.sidebar.title("🚀 Ladder Analytics")
page = st.sidebar.selectbox("Choose Dashboard", ["💸 Spending Feature", "📈 Signup Sources"])

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
        .metric-card {{
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {LADDER_BLUE};
            margin: 0.5rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SPENDING FEATURE DASHBOARD
# =============================================================================

if page == "💸 Spending Feature":
    
    # --- HEADER ---
    st.title("📊 Ladder Spending Feature - New Users Analysis")
    st.markdown("""
    This dashboard highlights new users and their interactions with the **Spending Feature** (Budgets & Transactions).
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
    def load_spending_data(start_date, end_date):
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
        start_date = end_date = date_range if hasattr(date_range, 'year') else min_date

    # Load data with selected date range
    with st.spinner("Loading filtered data from database..."):
        df = load_spending_data(start_date, end_date)

    if df.empty:
        st.warning("⚠️ No data available for the selected date range. Please try a different date range.")
    else:
        df['created_at'] = pd.to_datetime(df['created_at'])

        # Interaction filter
        interaction_filter = st.sidebar.selectbox("Interaction", ["All"] + df["interacted_with_spending_feature"].unique().tolist())

        # Apply interaction filter
        if interaction_filter != "All":
            df = df[df["interacted_with_spending_feature"] == interaction_filter]

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
                label="🧑‍💼 Total New Users", 
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
            legend_title="Interaction Status"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- FEATURE USAGE CHART ---
        st.subheader("💡 Spending Feature Usage Breakdown")
        feature_counts = df["spending_feature_used"].value_counts().reset_index()
        feature_counts.columns = ["Feature Type", "User Count"]

        fig_breakdown = px.pie(
            feature_counts, 
            values='User Count', 
            names='Feature Type',
            title='Distribution of Spending Feature Usage',
            color_discrete_sequence=['#0039A6', '#1347AD', '#FFA500', '#FF6B6B']
        )

        fig_breakdown.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_breakdown, use_container_width=True)

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
        search_term = st.text_input("🔍 Search users (name or email):", placeholder="Enter name or email to search...")

        if search_term:
            mask = (
                df['customer_name'].str.contains(search_term, case=False, na=False) |
                df['email'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = df[mask]
            st.info(f"Found {len(filtered_df)} users matching '{search_term}'")
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for Ladder ✨")

# =============================================================================
# SIGNUP SOURCES DASHBOARD
# =============================================================================

elif page == "📈 Signup Sources":
    
    # --- HEADER ---
    st.title("📈 Signup Sources Analysis")
    st.markdown("""
    This dashboard analyzes where users are signing up from based on questionnaire responses.
    """)

    # Only load OpenAI if API key is available
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            openai_available = True
        except ImportError:
            st.warning("⚠️ OpenAI library not available. Using simplified classification.")
            openai_available = False
    else:
        st.info("ℹ️ OpenAI API key not configured. Using simplified classification.")
        openai_available = False

    # --- EMBEDDING FUNCTION ---
    def get_embedding(text, model="text-embedding-3-small"):
        if not openai_available:
            return None
        try:
            text = text.replace("\n", " ")
            return client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None

    # --- FETCH SIGNUP SOURCES DATA ---
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_questionnaire_sources():
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT TRIM(elem->>'answer') AS answer
                FROM questionnaires,
                jsonb_array_elements(metadata::jsonb) AS elem
                WHERE elem->>'topic' = 'source'
                  AND DATE(created_at) BETWEEN DATE '2025-06-02' AND CURRENT_DATE
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [row[0] for row in rows if row[0]]
        except Exception as e:
            st.error(f"❌ Failed to fetch questionnaire data: {str(e)}")
            return []

    # --- CLASSIFICATION FUNCTION ---
    def classify_source_simple(answer, known_sources):
        """Simple classification without OpenAI"""
        answer_lower = answer.lower()
        for source in known_sources:
            if source.lower() in answer_lower or answer_lower in source.lower():
                return source
        return "Other/Unknown"

    def classify_source_ai(answer, known_embeddings, threshold=0.7):
        """AI-powered classification with OpenAI"""
        if not openai_available:
            return classify_source_simple(answer, list(known_embeddings.keys()))
        
        try:
            emb = get_embedding(answer)
            if emb is None:
                return classify_source_simple(answer, list(known_embeddings.keys()))
            
            similarities = {
                label: np.dot(emb, known_emb)
                for label, known_emb in known_embeddings.items()
                if known_emb is not None
            }
            
            if not similarities:
                return classify_source_simple(answer, list(known_embeddings.keys()))
                
            best_label, best_score = max(similarities.items(), key=lambda x: x[1])
            if best_score > threshold:
                return best_label
            else:
                return "Other/Unknown"
        except Exception as e:
            st.error(f"Error in AI classification: {e}")
            return classify_source_simple(answer, list(known_embeddings.keys()))

    # --- LOAD AND PROCESS DATA ---
    with st.spinner("Loading signup sources data..."):
        answers = fetch_questionnaire_sources()

    if not answers:
        st.warning("⚠️ No signup source data found for the specified date range.")
        st.stop()

    # Known sources
    known_sources = ["TikTok", "Janice", "Nubuke", "Instagram"]
    
    # Get embeddings if OpenAI is available
    known_embeddings = {}
    if openai_available:
        with st.spinner("Generating embeddings for known sources..."):
            for source in known_sources:
                emb = get_embedding(source)
                if emb is not None:
                    known_embeddings[source] = emb

    # Classify all answers
    results = {}
    classification_method = "AI-powered" if openai_available and known_embeddings else "Simple text matching"
    
    with st.spinner(f"Classifying signup sources using {classification_method}..."):
        for answer in answers:
            if openai_available and known_embeddings:
                label = classify_source_ai(answer, known_embeddings)
            else:
                label = classify_source_simple(answer, known_sources)
            results[label] = results.get(label, 0) + 1

    # Sort results by count
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # --- DISPLAY METRICS ---
    st.subheader("📊 Signup Sources Overview")
    
    total_responses = sum(results.values())
    st.info(f"**Total Responses Analyzed:** {total_responses:,} | **Classification Method:** {classification_method}")

    # Create metric cards in a grid
    num_sources = len(sorted_results)
    cols_per_row = 4
    
    for i in range(0, num_sources, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (source, count) in enumerate(sorted_results[i:i+cols_per_row]):
            with cols[j]:
                percentage = (count / total_responses * 100) if total_responses > 0 else 0
                
                # Choose emoji based on source
                emoji_map = {
                    "TikTok": "🎵",
                    "Instagram": "📷", 
                    "Janice": "👤",
                    "Nubuke": "👤",
                    "Other/Unknown": "❓"
                }
                emoji = emoji_map.get(source, "📈")
                
                st.metric(
                    label=f"{emoji} {source}",
                    value=f"{count:,}",
                    delta=f"{percentage:.1f}%"
                )
    
    pie_data = pd.DataFrame(sorted_results, columns=['Source', 'Count'])

    # --- BAR CHART ---
    st.subheader("📊 Source Comparison")
    
    fig_bar = px.bar(
        pie_data,
        x='Source',
        y='Count',
        title='Signup Count by Source',
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig_bar.update_layout(
        xaxis_title="Source",
        yaxis_title="Number of Signups",
        title_font_size=16
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- DETAILED TABLE ---
    st.subheader("📋 Detailed Source Breakdown")
    
    # Add percentage column to the dataframe
    pie_data['Percentage'] = (pie_data['Count'] / total_responses * 100).round(2)
    pie_data['Percentage_Formatted'] = pie_data['Percentage'].apply(lambda x: f"{x}%")
    
    st.dataframe(
        pie_data[['Source', 'Count', 'Percentage_Formatted']].rename(columns={'Percentage_Formatted': 'Percentage'}),
        use_container_width=True,
        hide_index=True
    )

    # --- SIDEBAR INFO ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About This Analysis**")
    st.sidebar.info(f"""
    • **Total Responses:** {total_responses:,}
    • **Date Range:** June 2, 2025 - Present
    • **Method:** {classification_method}
    • **Known Sources:** {len(known_sources)}
    """)
    
    st.sidebar.markdown("Built for Ladder ✨")
# --- END OF SIGNUP SOURCES DASHBOARD ---
else:
    st.error("❌ Invalid page selection. Please choose a valid dashboard from the sidebar.")



