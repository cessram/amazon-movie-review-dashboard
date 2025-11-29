"""
=============================================================================
AMAZON MOVIE REVIEWS - STREAMLIT EDA DASHBOARD
=============================================================================
Interactive dashboard to visualize EDA results.

Usage: streamlit run amazon-movie-streamlit-dashboard.py
Requirements: pip install streamlit plotly pymongo certifi python-dotenv
=============================================================================
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os

# Page config
st.set_page_config(
    page_title="Amazon Movie Reviews EDA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cinema dark theme
st.markdown("""
<style>
    /* Remove white header space */
    .stApp {
        background: #000000 !important;
    }
    header[data-testid="stHeader"] {
        background: #000000 !important;
    }
    .block-container {
        padding-top: 1rem !important;
        background: #000000 !important;
    }
    [data-testid="stToolbar"] {
        background: #000000 !important;
    }
    [data-testid="stDecoration"] {
        background: #000000 !important;
        display: none !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
        border-right: 1px solid #d4af37;
    }
    h1, h2, h3 { color: #d4af37 !important; text-shadow: 0 0 10px rgba(212, 175, 55, 0.3); }

    /* Make metric numbers more visible */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
        border: 2px solid #d4af37; border-radius: 10px; padding: 20px;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.3);
    }
    [data-testid="metric-container"] label,
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] > div > div {
    color: #d4af37 !important; 
    font-weight: bold !important;
    font-size: 1rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; 
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
        font-size: 2.2rem !important;
        font-weight: bold !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #0a0a0a; padding: 10px; border-radius: 10px; border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a1a; border-radius: 8px; color: #fff; border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #d4af37 0%, #b8960c 100%) !important;
        color: #000 !important; font-weight: bold;
    }
    .streamlit-expanderHeader {
        background: #0a0a0a !important; 
        border: 1px solid #d4af37; 
        border-radius: 8px; 
        color: #d4af37 !important;
    }

    /* Code block styling */
    .stCodeBlock, code, pre {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
        border: 1px solid #333 !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"], .stDataFrame {
        background: #0a0a0a !important;
    }
    [data-testid="stDataFrame"] div {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
    }
    .stDataFrame thead th {
        background: #1a1a1a !important;
        color: #d4af37 !important;
    }
    .stDataFrame tbody td {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
    }

    p, span, li { color: #e0e0e0 !important; }
    .stRadio label { color: #e0e0e0 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #d4af37; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def connect_mongodb():
    try:
        from pymongo import MongoClient
        import certifi
        from dotenv import load_dotenv
        load_dotenv()
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            return None
        client = MongoClient(uri, tlsCAFile=certifi.where())
        db = client[os.environ.get("MONGODB_DATABASE", "amazon_movies")]
        return db
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None


@st.cache_data
def load_stats_from_mongodb():
    db = connect_mongodb()
    if db is None:
        return None

    total_reviews = db.reviews.count_documents({})
    unique_users = db.user_stats.count_documents({})
    unique_products = db.product_stats.count_documents({})

    rating_results = list(db.reviews.aggregate([
        {"$group": {"_id": "$score", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]))
    rating_counts = {str(r['_id']): r['count'] for r in rating_results}

    yearly_results = list(db.reviews.aggregate([
        {"$group": {"_id": "$year", "count": {"$sum": 1}, "avg_rating": {"$avg": "$score"}}},
        {"$sort": {"_id": 1}}
    ]))
    yearly_data = {str(y['_id']): {'count': y['count'], 'avg_rating': y['avg_rating']}
                   for y in yearly_results if y['_id']}

    casual = db.user_stats.count_documents({"review_count": {"$lte": 5}})
    regular = db.user_stats.count_documents({"review_count": {"$gt": 5, "$lte": 50}})
    power = db.user_stats.count_documents({"review_count": {"$gt": 50}})

    top_products = list(db.product_stats.find().sort("review_count", -1).limit(15))

    avg_result = list(db.reviews.aggregate([
        {"$group": {"_id": None, "avg_rating": {"$avg": "$score"}, "avg_word_count": {"$avg": "$word_count"}}}
    ]))

    wc_results = list(db.reviews.aggregate([
        {"$group": {"_id": "$score", "avg_wc": {"$avg": "$word_count"}}},
        {"$sort": {"_id": 1}}
    ]))
    word_count_by_rating = {str(r['_id']): r['avg_wc'] for r in wc_results}

    help_results = list(db.reviews.aggregate([
        {"$match": {"helpful_ratio": {"$ne": None}}},
        {"$group": {"_id": "$score", "avg_helpful": {"$avg": "$helpful_ratio"}}},
        {"$sort": {"_id": 1}}
    ]))
    helpful_by_rating = {str(r['_id']): r['avg_helpful'] for r in help_results}

    return {
        'total_reviews': total_reviews,
        'unique_users': unique_users,
        'unique_products': unique_products,
        'avg_rating': avg_result[0]['avg_rating'] if avg_result else 0,
        'avg_word_count': avg_result[0]['avg_word_count'] if avg_result else 0,
        'rating_counts': rating_counts,
        'rating_percentages': {k: v / total_reviews * 100 for k, v in
                               rating_counts.items()} if total_reviews > 0 else {},
        'yearly_data': yearly_data,
        'user_segments': {'casual': casual, 'regular': regular, 'power': power},
        'power_users': power,
        'word_count_by_rating': word_count_by_rating,
        'helpful_by_rating': helpful_by_rating,
        'top_products': top_products,
        'source': 'MongoDB (150K Sample)'
    }


# =============================================================================
# DASHBOARD
# =============================================================================

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🎬 Amazon Movie Reviews</h1>
            <p style='color: #d4af37; font-size: 1.2rem; letter-spacing: 3px;'>
                ★ EXPLORATORY DATA ANALYSIS ★
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <p style='color: #d4af37; font-size: 1.5rem;'>🎬</p>
        <p style='color: #d4af37; letter-spacing: 2px; font-size: 0.8rem;'>CONTROL PANEL</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Load data
    with st.spinner("Loading data from MongoDB..."):
        stats = load_stats_from_mongodb()
        if stats is None:
            st.error("❌ Could not connect to MongoDB. Check your .env file.")
            return

    st.sidebar.success(f"✓ Loaded: {stats['source']}")

    # Key Metrics
    st.markdown("""
    <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 20px; margin: 20px 0;
                box-shadow: 0 0 25px rgba(212, 175, 55, 0.3);'>
        <h3 style='text-align: center; color: #ffd700; margin-bottom: 10px;'>🎬 DATASET OVERVIEW</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%); border: 2px solid #d4af37; 
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <p style='color: #d4af37; margin: 0; font-weight: bold;'>📊 Total Reviews</p>
            <p style='color: #ffffff; font-size: 2rem; font-weight: bold; margin: 0;'>{stats['total_reviews']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%); border: 2px solid #d4af37; 
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <p style='color: #d4af37; margin: 0; font-weight: bold;'>👥 Unique Users</p>
            <p style='color: #ffffff; font-size: 2rem; font-weight: bold; margin: 0;'>{stats['unique_users']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%); border: 2px solid #d4af37; 
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <p style='color: #d4af37; margin: 0; font-weight: bold;'>🎥 Products</p>
            <p style='color: #ffffff; font-size: 2rem; font-weight: bold; margin: 0;'>{stats['unique_products']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%); border: 2px solid #d4af37; 
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <p style='color: #d4af37; margin: 0; font-weight: bold;'>⭐ Avg Rating</p>
            <p style='color: #ffffff; font-size: 2rem; font-weight: bold; margin: 0;'>{stats['avg_rating']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # =================================================================
    # TABS
    # =================================================================
    tab0, tab7, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Dataset Info", "🔧 Methodology", "📈 Q1: Trends", "⭐ Q2: Ratings",
        "👍 Q3: Helpfulness", "📝 Q4: Text Analysis", "👥 Q5: Users", "🎥 Q6: Products"
    ])

    # -----------------------------------------------------------------
    # TAB 0: DATASET INFO
    # -----------------------------------------------------------------
    with tab0:
        st.markdown("<h2 style='text-align: center;'>📋 Dataset Information</h2>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 30px; margin: 20px 0;
                    box-shadow: 0 0 25px rgba(212, 175, 55, 0.3);'>
            <h3 style='color: #ffd700; text-align: center;'>🎬 Amazon Movie Reviews Dataset</h3>
            <p style='color: #ffffff; text-align: center; font-size: 1.1rem; margin-top: 15px;'>
                This dataset consists of movie reviews from Amazon. The data spans a period of more than 
                <span style='color: #ffd700; font-weight: bold;'>10 years</span>, including all 
                <span style='color: #ffd700; font-weight: bold;'>~8 million reviews</span> up to October 2012.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Statistics
        st.markdown("<h3 style='color: #d4af37;'>📊 Dataset Statistics</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #d4af37; 
                        border-radius: 10px; padding: 20px; text-align: center;'>
                <p style='color: #888; margin: 0;'>Total Reviews</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,215,0,0.5);'>7,911,684</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #d4af37; 
                        border-radius: 10px; padding: 20px; text-align: center;'>
                <p style='color: #888; margin: 0;'>Unique Users</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,215,0,0.5);'>889,176</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #d4af37; 
                        border-radius: 10px; padding: 20px; text-align: center;'>
                <p style='color: #888; margin: 0;'>Unique Products</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,215,0,0.5);'>253,059</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #ff6b35; 
                        border-radius: 10px; padding: 20px; text-align: center; margin-top: 15px;'>
                <p style='color: #888; margin: 0;'>Power Users (>50 reviews)</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,107,53,0.5);'>16,341</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #ff6b35; 
                        border-radius: 10px; padding: 20px; text-align: center; margin-top: 15px;'>
                <p style='color: #888; margin: 0;'>Median Words/Review</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,107,53,0.5);'>101</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: #0a0a0a; border: 2px solid #ff6b35; 
                        border-radius: 10px; padding: 20px; text-align: center; margin-top: 15px;'>
                <p style='color: #888; margin: 0;'>Timespan</p>
                <p style='color: #ffffff; font-size: 1.5rem; font-weight: bold; text-shadow: 0 0 10px rgba(255,107,53,0.5);'>Aug 1997 - Oct 2012</p>
            </div>
            """, unsafe_allow_html=True)

        # Data Format
        st.markdown("<h3 style='color: #d4af37; margin-top: 30px;'>📄 Data Format</h3>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #0a0a0a; border: 1px solid #d4af37; border-radius: 10px; 
                    padding: 20px; font-family: monospace; font-size: 0.9rem;'>
            <p><span style='color: #d4af37;'>product/productId:</span> <span style='color: #ffffff;'>B00006HAXW</span></p>
            <p><span style='color: #d4af37;'>review/userId:</span> <span style='color: #ffffff;'>A1RSDE90N6RSZF</span></p>
            <p><span style='color: #d4af37;'>review/profileName:</span> <span style='color: #ffffff;'>Joseph M. Kotow</span></p>
            <p><span style='color: #d4af37;'>review/helpfulness:</span> <span style='color: #ffffff;'>9/9</span></p>
            <p><span style='color: #d4af37;'>review/score:</span> <span style='color: #ffffff;'>5.0</span></p>
            <p><span style='color: #d4af37;'>review/time:</span> <span style='color: #ffffff;'>1042502400</span></p>
            <p><span style='color: #d4af37;'>review/summary:</span> <span style='color: #ffffff;'>Pittsburgh - Home of the OLDIES</span></p>
            <p><span style='color: #d4af37;'>review/text:</span> <span style='color: #ffffff;'>I have all of the doo wop DVD's...</span></p>
        </div>
        """, unsafe_allow_html=True)

        # Field descriptions
        st.markdown("<h4 style='color: #d4af37; margin-top: 25px;'>📝 Field Descriptions</h4>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #0a0a0a; border: 1px solid #333; border-radius: 10px; overflow: hidden;'>
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr style='background: #1a1a1a;'>
                        <th style='padding: 12px 15px; text-align: left; color: #d4af37; border-bottom: 2px solid #d4af37;'>Field</th>
                        <th style='padding: 12px 15px; text-align: left; color: #d4af37; border-bottom: 2px solid #d4af37;'>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>product/productId</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Amazon Standard Identification Number (ASIN)</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333; background: #0d0d0d;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/userId</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Unique identifier for the reviewer</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/profileName</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Display name of the reviewer</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333; background: #0d0d0d;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/helpfulness</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Fraction of users who found review helpful</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/score</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Rating (1.0 to 5.0)</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333; background: #0d0d0d;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/time</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Unix timestamp</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/summary</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Brief summary/title</td>
                    </tr>
                    <tr style='background: #0d0d0d;'>
                        <td style='padding: 10px 15px; color: #ffd700;'>review/text</td>
                        <td style='padding: 10px 15px; color: #ffffff;'>Full review text</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Citation
        st.markdown("""
        <div style='background: #0a0a0a; border-left: 4px solid #d4af37; padding: 20px; margin: 20px 0;'>
            <p style='color: #ffd700; font-weight: bold;'>📚 Citation</p>
            <p style='color: #ffffff; font-style: italic;'>
                J. McAuley and J. Leskovec. "From amateurs to connoisseurs: modeling the evolution 
                of user expertise through online reviews." WWW, 2013.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 7: METHODOLOGY
    # -----------------------------------------------------------------
    with tab7:
        st.markdown("<h2 style='text-align: center;'>🔧 Project Methodology</h2>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 30px; margin: 20px 0;
                    box-shadow: 0 0 25px rgba(212, 175, 55, 0.3);'>
            <h3 style='color: #ffd700; text-align: center;'>🏗️ AWS Cloud Data Pipeline</h3>
            <p style='color: #ffffff; text-align: center;'>
                End-to-end data engineering pipeline using AWS services and MongoDB for analyzing 8 million Amazon movie reviews.
            </p>
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "☁️", "#ff9900", "AWS Infrastructure Setup", "aws-creation-boto3.py",
             "Programmatically provision AWS resources using Python and Boto3",
             ["Create S3 bucket for data storage", "Configure IAM roles and policies", "Set up security groups"]),
            ("2", "🍃", "#00ed64", "MongoDB Atlas Setup", "MongoDB Atlas Console",
             "Set up cloud-hosted NoSQL database for storing processed reviews",
             ["Create MongoDB Atlas free tier cluster", "Configure network access", "Set up database credentials"]),
            ("3", "🔄", "#d4af37", "Data Pipeline & ETL", "amazon-s3load-mongodb.py",
             "Download, process, and load raw data into cloud storage",
             ["Download raw data from Stanford", "Convert to Parquet format", "Upload to S3",
              "Load 150K sample to MongoDB"]),
            ("4", "📊", "#c41e3a", "Exploratory Data Analysis", "mongodb-eda.py",
             "Compute comprehensive statistics from MongoDB",
             ["Calculate rating distributions", "Analyze user behavior", "Generate visualizations"]),
            ("5", "🎬", "#ff4b4b", "Interactive Dashboard", "amazon-movie-streamlit-dashboard.py",
             "Create interactive web dashboard for data exploration",
             ["Connect to MongoDB", "Build Plotly visualizations", "Deploy with Streamlit"])
        ]

        for num, icon, color, title, file, purpose, tasks in steps:
            tasks_html = "".join(f"<li>{t}</li>" for t in tasks)
            st.markdown(f"""
            <div style='background: #0a0a0a; border-left: 4px solid {color}; border-radius: 0 10px 10px 0; 
                        padding: 25px; margin: 15px 0; border: 1px solid #333;'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span style='font-size: 2rem; margin-right: 15px;'>{icon}</span>
                    <div>
                        <p style='color: {color}; font-size: 0.9rem; margin: 0;'>STEP {num}</p>
                        <h4 style='color: #ffffff; margin: 5px 0;'>{title}</h4>
                    </div>
                </div>
                <p style='color: #888; font-family: monospace; background: #000; padding: 5px 10px; 
                          border-radius: 5px; display: inline-block; margin-bottom: 10px; border: 1px solid #333;'>📁 {file}</p>
                <p style='color: #ffd700;'><strong>Purpose:</strong> <span style='color: #ffffff;'>{purpose}</span></p>
                <ul style='color: #e0e0e0; margin: 0; padding-left: 20px;'>{tasks_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # Architecture
        st.markdown("<h3 style='color: #d4af37; text-align: center; margin-top: 30px;'>🏛️ Architecture Overview</h3>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 10px; padding: 30px; text-align: center;'>
            <p style='font-size: 1.2rem;'>
                <span style='color: #ffd700;'>📥 Raw Data</span>
                <span style='color: #666;'> → </span>
                <span style='color: #ff9900;'>☁️ S3 Bucket</span>
                <span style='color: #666;'> → </span>
                <span style='color: #00ed64;'>🍃 MongoDB</span>
                <span style='color: #666;'> → </span>
                <span style='color: #ff4b4b;'>🎬 Streamlit</span>
            </p>
            <p style='color: #ffffff; font-size: 0.9rem; margin-top: 15px;'>8M Reviews → Parquet Chunks → 150K Sample → Interactive Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        # Tech Stack
        st.markdown("<h3 style='color: #d4af37; text-align: center; margin-top: 30px;'>🛠️ Technology Stack</h3>",
                    unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        for col, (emoji, name, color, desc) in zip([col1, col2, col3, col4],
                                                   [("🐍", "Python", "#ffd700", "Core Language"),
                                                    ("☁️", "AWS S3", "#ff9900", "Data Lake"),
                                                    ("🍃", "MongoDB", "#00ed64", "NoSQL DB"),
                                                    ("📊", "Streamlit", "#ff4b4b", "Dashboard")]):
            with col:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: #0a0a0a; 
                            border: 2px solid {color}; border-radius: 10px;'>
                    <p style='font-size: 2.5rem;'>{emoji}</p>
                    <p style='color: {color}; font-weight: bold;'>{name}</p>
                    <p style='color: #ffffff; font-size: 0.8rem;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 1: TRENDS
    # -----------------------------------------------------------------
    with tab1:
        st.markdown("<h2 style='text-align: center;'>🎞️ Rating & Volume Trends Over Time</h2>", unsafe_allow_html=True)

        years = sorted([int(y) for y in stats['yearly_data'].keys() if int(y) >= 1997])
        counts = [stats['yearly_data'][str(y)]['count'] for y in years]
        avg_ratings = [stats['yearly_data'][str(y)]['avg_rating'] for y in years]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=years, y=counts, name="Review Count",
                             marker=dict(color='#d4af37', line=dict(color='#ffd700', width=1.5)),
                             opacity=0.9), secondary_y=False)
        fig.add_trace(go.Scatter(x=years, y=avg_ratings, name="Avg Rating",
                                 line=dict(color='#ff4444', width=4),
                                 mode='lines+markers',
                                 marker=dict(size=12, color='#ff4444', line=dict(color='#ffd700', width=2))),
                      secondary_y=True)
        fig.update_layout(title=dict(text="📊 Reviews & Ratings Over Time", font=dict(color='#d4af37', size=20)),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.95)',
                          height=500, font=dict(color='#e0e0e0'),
                          legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='#ffd700')))
        fig.update_xaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
        fig.update_yaxes(title_text="Reviews", secondary_y=False, gridcolor='#2a2a2a',
                         tickfont=dict(color='#d4af37'), title_font=dict(color='#d4af37'))
        fig.update_yaxes(title_text="Avg Rating", secondary_y=True, range=[3.5, 5],
                         gridcolor='#2a2a2a', tickfont=dict(color='#ff4444'), title_font=dict(color='#ff4444'))
        st.plotly_chart(fig, use_container_width=True)

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Explosive Growth:</strong> Review volume increased dramatically from 1997 to 2012</li>
                <li><strong style='color: #ffd700;'>Rating Stability:</strong> Average ratings remained consistently high (4.0-4.5★) across all years</li>
                <li><strong style='color: #ffd700;'>Platform Maturity:</strong> Steady increase indicates growing user adoption</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 2: RATINGS
    # -----------------------------------------------------------------
    with tab2:
        st.markdown("<h2 style='text-align: center;'>⭐ Rating Distribution</h2>", unsafe_allow_html=True)

        ratings = ['1.0', '2.0', '3.0', '4.0', '5.0']
        counts = [stats['rating_counts'].get(r, 0) for r in ratings]
        colors = ['#ff3333', '#ff6b35', '#ffa500', '#ffd700', '#ffec00']

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Bar(x=[f"{int(float(r))}★" for r in ratings], y=counts,
                                         marker=dict(color=colors, line=dict(color='#ffffff', width=2)),
                                         text=[f'{c:,}' for c in counts], textposition='outside',
                                         textfont=dict(color='#ffd700', size=12))])
            fig.update_layout(title=dict(text="🎭 Rating Distribution", font=dict(color='#d4af37', size=18)),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.95)',
                              font=dict(color='#e0e0e0'), height=400)
            fig.update_xaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
            fig.update_yaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(data=[go.Pie(labels=[f"{int(float(r))}★" for r in ratings], values=counts,
                                         marker=dict(colors=colors, line=dict(color='#1a1a1a', width=3)),
                                         hole=0.45, textfont=dict(color='#000000', size=14))])
            fig.update_layout(title=dict(text="🎬 Rating Proportions", font=dict(color='#d4af37', size=18)),
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'), height=400,
                              legend=dict(font=dict(color='#ffd700')),
                              annotations=[dict(text='Ratings', x=0.5, y=0.5, font_size=18,
                                                font_color='#d4af37', showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Positive Skew:</strong> Majority of reviews are 5-star ratings</li>
                <li><strong style='color: #ffd700;'>J-Shaped Distribution:</strong> High frequencies at extremes (1★ and 5★)</li>
                <li><strong style='color: #ffd700;'>Selection Bias:</strong> Users more likely to review when very satisfied or dissatisfied</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 3: HELPFULNESS
    # -----------------------------------------------------------------
    with tab3:
        st.markdown("<h2 style='text-align: center;'>👍 Helpfulness Analysis</h2>", unsafe_allow_html=True)

        ratings = ['1.0', '2.0', '3.0', '4.0', '5.0']
        helpful_vals = [stats['helpful_by_rating'].get(r, 0) for r in ratings]
        colors = ['#ff3333', '#ff6b35', '#ffa500', '#ffd700', '#ffec00']

        fig = go.Figure(data=[go.Bar(x=[f"{int(float(r))}★" for r in ratings], y=helpful_vals,
                                     marker=dict(color=colors, line=dict(color='#ffffff', width=2)),
                                     text=[f'{h:.2f}' for h in helpful_vals], textposition='outside',
                                     textfont=dict(color='#ffd700', size=14))])
        fig.update_layout(title=dict(text="🎯 Average Helpfulness Ratio by Rating", font=dict(color='#d4af37', size=18)),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.95)',
                          font=dict(color='#e0e0e0'), height=500)
        fig.update_xaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
        fig.update_yaxes(range=[0, 1], gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
        st.plotly_chart(fig, use_container_width=True)

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Critical Reviews Win:</strong> Lower-rated reviews tend to have higher helpfulness ratios</li>
                <li><strong style='color: #ffd700;'>Information Value:</strong> Negative reviews provide more detailed reasoning</li>
                <li><strong style='color: #ffd700;'>User Behavior:</strong> Readers seek out critical reviews for purchase decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 4: TEXT ANALYSIS
    # -----------------------------------------------------------------
    with tab4:
        st.markdown("<h2 style='text-align: center;'>📝 Review Text Characteristics</h2>", unsafe_allow_html=True)

        ratings = ['1.0', '2.0', '3.0', '4.0', '5.0']
        word_counts = [stats['word_count_by_rating'].get(r, 0) for r in ratings]
        colors = ['#ff3333', '#ff6b35', '#ffa500', '#ffd700', '#ffec00']

        fig = go.Figure(data=[go.Bar(x=[f"{int(float(r))}★" for r in ratings], y=word_counts,
                                     marker=dict(color=colors, line=dict(color='#ffffff', width=2)),
                                     text=[f'{w:.0f}' for w in word_counts], textposition='outside',
                                     textfont=dict(color='#ffd700', size=14))])
        fig.update_layout(title=dict(text="📖 Average Word Count by Rating", font=dict(color='#d4af37', size=18)),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.95)',
                          font=dict(color='#e0e0e0'), height=500)
        fig.update_xaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
        fig.update_yaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Avg Word Count", f"{stats['avg_word_count']:.0f} words")
        with col2:
            if word_counts:
                longest = max(word_counts)
                longest_idx = word_counts.index(longest)
                st.metric("📖 Longest Reviews", f"{int(float(ratings[longest_idx]))}★ ({longest:.0f} words)")

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Negative = Verbose:</strong> Lower-rated reviews tend to be longer</li>
                <li><strong style='color: #ffd700;'>Quick Praise:</strong> 5-star reviews are often shorter</li>
                <li><strong style='color: #ffd700;'>Median Length:</strong> ~101 words suggests concise but substantive reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 5: USERS
    # -----------------------------------------------------------------
    with tab5:
        st.markdown("<h2 style='text-align: center;'>👥 User Activity Analysis</h2>", unsafe_allow_html=True)

        segments = stats['user_segments']
        total_users = sum(segments.values()) if sum(segments.values()) > 0 else 1

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['🎫 Casual (1-5)', '🎟️ Regular (6-50)', '🏆 Power (>50)'],
                values=[segments['casual'], segments['regular'], segments['power']],
                marker=dict(colors=['#ff6b35', '#ffd700', '#ffec00'], line=dict(color='#1a1a1a', width=3)),
                hole=0.45, textfont=dict(color='#000000', size=12))])
            fig.update_layout(title=dict(text="🎭 User Segments", font=dict(color='#d4af37', size=18)),
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'), height=400,
                              legend=dict(font=dict(color='#ffd700')),
                              annotations=[dict(text='Users', x=0.5, y=0.5, font_size=18,
                                                font_color='#d4af37', showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h3 style='color: #d4af37;'>📊 Segment Breakdown</h3>", unsafe_allow_html=True)
            for name, key, color in [("🎫 Casual (1-5)", 'casual', '#ff6b35'),
                                     ("🎟️ Regular (6-50)", 'regular', '#ffd700'),
                                     ("🏆 Power (>50)", 'power', '#ffec00')]:
                st.markdown(f"""
                <div style='background: #0a0a0a; border: 2px solid {color}; 
                            border-radius: 10px; padding: 15px; margin: 10px 0;'>
                    <p style='color: {color}; font-weight: bold; margin-bottom: 5px;'>{name}</p>
                    <p style='color: #ffffff; font-size: 1.5rem; font-weight: bold;'>{segments[key]:,} 
                       <span style='font-size: 1rem; color: #888;'>({segments[key] / total_users * 100:.1f}%)</span></p>
                </div>
                """, unsafe_allow_html=True)

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Long-Tail Distribution:</strong> Majority of users write only a few reviews</li>
                <li><strong style='color: #ffd700;'>Power Law:</strong> Small group of power users contributes disproportionately</li>
                <li><strong style='color: #ffd700;'>Community Builders:</strong> ~16,341 power users are the backbone of the ecosystem</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 6: PRODUCTS
    # -----------------------------------------------------------------
    with tab6:
        st.markdown("<h2 style='text-align: center;'>🎥 Product Analysis</h2>", unsafe_allow_html=True)

        if stats['top_products']:
            products = stats['top_products']
            product_ids = [p['_id'][:15] + '...' for p in products]
            review_counts = [p['review_count'] for p in products]
            avg_ratings = [p['avg_rating'] for p in products]

            fig = go.Figure(data=[go.Bar(y=product_ids, x=review_counts, orientation='h',
                                         marker=dict(color=avg_ratings,
                                                     colorscale=[[0, '#ff3333'], [0.5, '#ffa500'], [1, '#ffec00']],
                                                     line=dict(color='#ffffff', width=1.5),
                                                     colorbar=dict(
                                                         title=dict(text='Rating', font=dict(color='#d4af37')),
                                                         tickfont=dict(color='#ffd700'),
                                                         bgcolor='rgba(0,0,0,0.5)')),
                                         text=[f'{c:,} ({r:.1f}★)' for c, r in zip(review_counts, avg_ratings)],
                                         textposition='outside', textfont=dict(color='#ffd700', size=11))])
            fig.update_layout(title=dict(text="🏆 Top 15 Most Reviewed Products", font=dict(color='#d4af37', size=18)),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.95)',
                              font=dict(color='#e0e0e0'), height=600, yaxis=dict(autorange='reversed'))
            fig.update_xaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
            fig.update_yaxes(gridcolor='#2a2a2a', linecolor='#d4af37', tickfont=dict(color='#ffd700'))
            st.plotly_chart(fig, use_container_width=True)

        # Key Findings
        st.markdown("""
        <div style='background: #0a0a0a; border: 2px solid #d4af37; border-radius: 15px; padding: 25px; margin: 20px 0;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.2);'>
            <h3 style='color: #d4af37; margin-bottom: 15px;'>🔍 Key Findings</h3>
            <ul style='color: #ffffff; line-height: 2.2;'>
                <li><strong style='color: #ffd700;'>Popular ≠ Highest Rated:</strong> Most-reviewed products don't always have highest ratings</li>
                <li><strong style='color: #ffd700;'>Blockbuster Effect:</strong> Top products receive significantly more reviews than average</li>
                <li><strong style='color: #ffd700;'>253,059 Products:</strong> Vast catalog with highly varied review counts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # =================================================================
    # FOOTER
    # =================================================================
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-top: 30px; 
                background: #0a0a0a; border-top: 2px solid #d4af37;'>
        <p style='color: #ffd700; font-size: 1.5rem;'>★ ★ ★ ★ ★</p>
        <p style='color: #d4af37; font-size: 1.1rem;'>ALY6110 Final Project | Amazon Movie Reviews EDA</p>
        <p style='color: #888; font-size: 0.9rem;'>🎬 Crafted with Streamlit & Plotly 🎬</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()