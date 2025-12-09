"""
=============================================================================
AMAZON MOVIE REVIEWS - BIG DATA ANALYTICS DASHBOARD
=============================================================================
ALY6110 Final Project - Cloud & Big Data Management

Dataset: Stanford SNAP Amazon Movie Reviews (~8 million reviews)
Pipeline: S3 (Boto3) → AWS Glue (PySpark) → Parquet → Spark SQL → MongoDB → Streamlit

Usage: streamlit run amazon-movie-dashboard-v2.py
=============================================================================
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os


# =============================================================================
# LOAD MONGODB CREDENTIALS
# =============================================================================
try:
    from config import MONGODB_URI, MONGODB_DATABASE
except ImportError:
    from dotenv import load_dotenv
    import pathlib
    script_dir = pathlib.Path(__file__).parent.resolve()
    load_dotenv(script_dir / '.env', override=True)
    MONGODB_URI = os.environ.get("MONGODB_URI")
    MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "amazon_movies")

# =============================================================================
# COLOR SCHEME - 5 Colors Only
# =============================================================================
GOLD = "#d4af37"
WHITE = "#e0e0e0"
PURPLE = "#9b59b6"
CORAL = "#e74c3c"
ORANGE = "#f39c12"

# Page config
st.set_page_config(
    page_title="Amazon Movie Reviews - Big Data Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS with 5-color scheme
st.markdown(f"""
<style>
    .stApp {{ background: #000000 !important; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{ padding-top: 1rem !important; }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
        border-right: 1px solid {GOLD};
    }}
    
    h1, h2 {{ color: {GOLD} !important; }}
    h3 {{ color: {ORANGE} !important; }}
    h4 {{ color: {WHITE} !important; }}
    
    .stTabs [data-baseweb="tab-list"] {{ 
        gap: 8px; background: #0a0a0a; padding: 10px; border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{ 
        background: #1a1a1a; border-radius: 8px; color: {WHITE}; 
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {GOLD} 0%, {ORANGE} 100%) !important;
        color: #000 !important; font-weight: bold;
    }}
    
    p, span, li, td, th, label {{ color: {WHITE} !important; }}
    strong {{ color: {GOLD} !important; }}
    a {{ color: {ORANGE} !important; }}
    code {{ color: {ORANGE} !important; background: #1a1a1a !important; }}
    
    #MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MONGODB CONNECTION
# =============================================================================
def get_mongodb_connection():
    from pymongo import MongoClient
    import certifi
    if not MONGODB_URI:
        return None
    try:
        client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000)
        client.admin.command('ping')
        return client[MONGODB_DATABASE]
    except Exception as e:
        st.error(f"MongoDB connection error: {e}")
        return None


@st.cache_data(ttl=3600)
def load_spark_sql_results():
    db = get_mongodb_connection()
    if db is None:
        return None
    try:
        return {
            'yearly_stats': list(db.yearly_stats.find().sort("_id", 1)),
            'rating_distribution': list(db.rating_distribution.find().sort("_id", 1)),
            'helpfulness_stats': list(db.helpfulness_stats.find().sort("_id", 1)),
            'user_segments': list(db.user_segments.find()),
            'product_stats': list(db.product_stats.find().sort("review_count", -1).limit(15)),
            'monthly_patterns': list(db.monthly_patterns.find().sort("_id", 1)),
            'total_reviews_sample': db.reviews.count_documents({}),
            'total_reviews_full': 7911684,
            'unique_users': 889176,
            'unique_products': 253059,
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# =============================================================================
# MERMAID DIAGRAM - Fixed for Streamlit
# =============================================================================
def render_pipeline_diagram():
    """Render pipeline diagram using st.components.v1.html for reliable rendering."""
    import streamlit.components.v1 as components

    pipeline_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                background: #0a0a0a; 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 20px;
            }
            .container {
                background: #0a0a0a; 
                padding: 30px; 
                border-radius: 15px; 
                border: 2px solid #d4af37;
            }
            .row {
                display: flex; 
                justify-content: center; 
                margin-bottom: 15px;
            }
            .arrow {
                text-align: center; 
                color: #d4af37; 
                font-size: 1.5rem; 
                margin: 10px 0;
            }
            .box {
                padding: 15px 25px; 
                border-radius: 10px; 
                text-align: center; 
                min-width: 200px;
            }
            .box-icon { font-size: 1.5rem; }
            .box-title { font-weight: bold; margin-top: 5px; }
            .box-subtitle { font-size: 0.85rem; margin-top: 3px; }
            .section {
                background: #1a1a1a; 
                border-radius: 15px; 
                padding: 20px; 
                margin-bottom: 15px;
            }
            .section-title {
                text-align: center; 
                font-weight: bold; 
                margin-bottom: 15px;
            }
            .flex-wrap {
                display: flex; 
                justify-content: center; 
                gap: 15px; 
                flex-wrap: wrap;
            }
            .small-box {
                padding: 12px 20px; 
                border-radius: 8px; 
                text-align: center;
            }
            .small-box-icon { font-size: 1rem; }
            .small-box-text { font-size: 0.85rem; margin-top: 3px; }
            .legend {
                margin-top: 25px; 
                padding-top: 20px; 
                border-top: 1px solid #333;
            }
            .legend-title {
                text-align: center; 
                color: #e0e0e0; 
                font-size: 0.9rem; 
                margin-bottom: 10px;
            }
            .legend-items {
                display: flex; 
                justify-content: center; 
                gap: 20px; 
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex; 
                align-items: center; 
                gap: 8px;
            }
            .legend-color {
                width: 20px; 
                height: 20px; 
                border-radius: 4px;
            }
            .legend-text {
                color: #e0e0e0; 
                font-size: 0.85rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Stanford SNAP -->
            <div class="row">
                <div class="box" style="background: #9b59b6; color: white;">
                    <div class="box-icon">📦</div>
                    <div class="box-title">Stanford SNAP</div>
                    <div class="box-subtitle">8M Movie Reviews , ~3GB</div>
                </div>
            </div>

            <div class="arrow">▼</div>

            <!-- AWS Infrastructure -->
            <div class="section" style="border: 2px solid #f39c12;">
                <div class="section-title" style="color: #f39c12;">☁️ AWS Infrastructure (Boto3)</div>
                <div class="flex-wrap">
                    <div class="small-box" style="background: #f39c12; color: black;">
                        <div class="small-box-icon">🔐</div>
                        <div class="small-box-text">IAM Role</div>
                    </div>
                    <div class="small-box" style="background: #f39c12; color: black;">
                        <div class="small-box-icon">📁</div>
                        <div class="small-box-text">S3: raw/</div>
                    </div>
                    <div class="small-box" style="background: #f39c12; color: black;">
                        <div class="small-box-icon">📁</div>
                        <div class="small-box-text">S3: processed/</div>
                    </div>
                    <div class="small-box" style="background: #f39c12; color: black;">
                        <div class="small-box-icon">📊</div>
                        <div class="small-box-text">S3: analytics/</div>
                    </div>
                </div>
            </div>

            <div class="arrow">▼</div>

            <!-- AWS Glue -->
            <div class="section" style="border: 2px solid #9b59b6;">
                <div class="section-title" style="color: #9b59b6;">⚡ AWS Glue (PySpark ETL)</div>
                <div class="flex-wrap">
                    <div class="small-box" style="background: #9b59b6; color: white;">
                        <div class="small-box-icon">🗄️</div>
                        <div class="small-box-text">Glue Catalog</div>
                    </div>
                    <div class="small-box" style="background: #9b59b6; color: white;">
                        <div class="small-box-icon">🔧</div>
                        <div class="small-box-text">PySpark Job</div>
                    </div>
                </div>
            </div>

            <div class="arrow">▼</div>

            <!-- Spark SQL -->
            <div class="section" style="border: 2px solid #e74c3c;">
                <div class="section-title" style="color: #e74c3c;">🔥 Spark SQL Analytics</div>
                <div class="flex-wrap" style="gap: 10px;">
                    <div class="small-box" style="background: #e74c3c; color: white; padding: 10px 15px;">
                        <div class="small-box-text">Q1: Yearly Trends</div>
                    </div>
                    <div class="small-box" style="background: #e74c3c; color: white; padding: 10px 15px;">
                        <div class="small-box-text">Q2: Ratings</div>
                    </div>
                    <div class="small-box" style="background: #e74c3c; color: white; padding: 10px 15px;">
                        <div class="small-box-text">Q3: Helpfulness</div>
                    </div>
                    <div class="small-box" style="background: #e74c3c; color: white; padding: 10px 15px;">
                        <div class="small-box-text">Q4: Users</div>
                    </div>
                    <div class="small-box" style="background: #e74c3c; color: white; padding: 10px 15px;">
                        <div class="small-box-text">Q5: Products</div>
                    </div>
                </div>
            </div>

            <div class="arrow">▼</div>

            <!-- MongoDB -->
            <div class="row">
                <div class="box" style="background: #d4af37; color: black;">
                    <div class="box-icon">🍃</div>
                    <div class="box-title">MongoDB Atlas</div>
                    <div class="box-subtitle">Pre-computed Results</div>
                </div>
            </div>

            <div class="arrow">▼</div>

            <!-- Streamlit -->
            <div class="row">
                <div class="box" style="background: #d4af37; color: black;">
                    <div class="box-icon">🎬</div>
                    <div class="box-title">Streamlit Dashboard</div>
                    <div class="box-subtitle">Interactive Visualizations</div>
                </div>
            </div>

            <!-- Legend -->
            <div class="legend">
                <div class="legend-title">Legend:</div>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #9b59b6;"></div>
                        <span class="legend-text">Data Source / Glue</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f39c12;"></div>
                        <span class="legend-text">AWS S3 (Boto3)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #e74c3c;"></div>
                        <span class="legend-text">Spark SQL</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #d4af37;"></div>
                        <span class="legend-text">MongoDB / Dashboard</span>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(pipeline_html, height=1250, scrolling=False)


# =============================================================================
# CODE SNIPPETS FOR METHODOLOGY
# =============================================================================
CODE_STEP1 = '''# Environment Setup - .env file
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=amazon-movie-reviews-aly6110
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/'''

CODE_STEP2 = '''# AWS S3 Infrastructure with Boto3
def create_s3_bucket(s3_client, bucket_name, region):
    """Create S3 bucket with folder structure."""
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create folder structure for data lake
    for folder in ['raw/', 'staged/', 'processed/', 'analytics/']:
        s3_client.put_object(Bucket=bucket_name, Key=folder, Body='')
    
    # Apply project tags
    s3_client.put_bucket_tagging(
        Bucket=bucket_name,
        Tagging={'TagSet': [{'Key': 'project', 'Value': 'aly6110'}]}
    )
    print(f"✓ S3 Bucket '{bucket_name}' created with tags")'''

CODE_STEP3 = '''# Upload Raw Data to S3
def upload_raw_data_to_s3(s3_client, local_path, bucket_name):
    """Upload raw compressed file to S3 raw/ prefix."""
    s3_key = "raw/" + os.path.basename(local_path)
    s3_client.upload_file(local_path, bucket_name, s3_key)
    print(f"✓ Uploaded to s3://{bucket_name}/{s3_key}")'''

CODE_STEP4 = '''# AWS Glue ETL Job with PySpark
def create_glue_job(glue_client, role_arn, script_location):
    """Create serverless Glue ETL job."""
    glue_client.create_job(
        Name='amazon-movies-etl-job',
        Role=role_arn,
        Command={'Name': 'glueetl', 'ScriptLocation': script_location},
        GlueVersion='4.0',
        NumberOfWorkers=5,
        WorkerType='G.1X'  # Distributed processing
    )'''

CODE_SPARK_SQL = '''# Spark SQL Analytics (Inside Glue Job)
df_transformed.createOrReplaceTempView("reviews")

# Q1: Yearly Trends - Aggregation at scale
yearly_stats = spark.sql("""
    SELECT year, COUNT(*) as review_count,
           ROUND(AVG(score), 2) as avg_rating,
           COUNT(DISTINCT user_id) as unique_reviewers
    FROM reviews WHERE year BETWEEN 1997 AND 2012
    GROUP BY year ORDER BY year
""")

# Q2: Rating Distribution
rating_dist = spark.sql("""
    SELECT score, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM reviews GROUP BY score ORDER BY score
""")

# Save results to S3 as Parquet (columnar format)
yearly_stats.write.mode("overwrite").parquet("s3://bucket/analytics/yearly/")'''

CODE_STEP7 = '''# Load Pre-computed Results to MongoDB
def load_to_mongodb(db, s3_client):
    """Load Spark SQL results to MongoDB for dashboard."""
    # Load yearly stats from S3 Parquet
    df = read_parquet_from_s3(s3_client, bucket, "analytics/yearly/")
    
    # Insert to MongoDB collection
    documents = df.to_dict('records')
    db['yearly_stats'].delete_many({})
    db['yearly_stats'].insert_many(documents)
    
    # Create indexes for fast queries
    db['yearly_stats'].create_index([("year", 1)])
    print(f"✓ Loaded {len(documents)} yearly stats to MongoDB")'''


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    # Header
    st.markdown(f"""
        <div style='text-align: center; padding: 25px 0; border-bottom: 2px solid {GOLD};'>
            <h1 style='font-size: 2.5rem; margin-bottom: 5px;'>🎬 Amazon Movie Reviews</h1>
            <h2 style='font-size: 1.5rem; color: {WHITE} !important; font-weight: normal;'>Big Data Analytics Dashboard</h2>
            <p style='color: {ORANGE}; font-size: 1rem; margin-top: 10px;'>
                ALY6110 - Cloud & Big Data Management | Final Project
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(f"<h3>📊 Project Info</h3>", unsafe_allow_html=True)
    st.sidebar.info("**Course:** ALY6110\n\n**Dataset:** 8M Amazon Reviews\n\n**Tools:** AWS S3, Glue, PySpark, Spark SQL, MongoDB")

    # Load data
    with st.spinner("Loading data from MongoDB..."):
        stats = load_spark_sql_results()
        if stats is None:
            st.error("❌ Could not connect to MongoDB.")
            return

    st.sidebar.success(f"✓ Connected\n\n📊 {stats['total_reviews_sample']:,} sampled reviews")

    # =================================================================
    # TABS - Aligned with Rubric
    # =================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Introduction",
        "🔧 Methodology",
        "📊 Results & Visualizations",
        "📋 Conclusions"
    ])

    # -----------------------------------------------------------------
    # TAB 1: INTRODUCTION (25 pts)
    # Real-world problem + Dataset + Connection
    # -----------------------------------------------------------------
    with tab1:
        st.markdown(f"<h2>📌 Introduction: The Real-World Problem</h2>", unsafe_allow_html=True)

        # Problem Statement
        st.markdown(f"""
        <div style='background: #111; border: 2px solid {GOLD}; border-radius: 15px; padding: 30px; margin: 20px 0;'>
            <h3 style='text-align: center;'>🎯 The Big Data Challenge</h3>
            <p style='font-size: 1.1rem; line-height: 1.9; margin-top: 20px;'>
                In today's digital economy, <strong>streaming platforms</strong> (Netflix, Amazon Prime, Disney+) 
                and <strong>e-commerce companies</strong> generate massive volumes of customer review data daily. 
                These companies face a critical challenge:
            </p>
            <blockquote style='border-left: 4px solid {ORANGE}; padding: 20px; margin: 25px 0; background: #0a0a0a;'>
                <p style='font-size: 1.15rem; font-style: italic; color: {ORANGE} !important;'>
                    "How can we process and analyze millions of customer reviews at scale to understand 
                    rating patterns, identify helpful content, and segment user behavior?"
                </p>
            </blockquote>
            <p style='font-size: 1.05rem; line-height: 1.8;'>
                Traditional data processing tools <strong>cannot handle this scale</strong>. With 
                <strong>8 million reviews</strong> generating gigabytes of text data, businesses require 
                <strong>cloud-based distributed computing solutions</strong> to extract meaningful insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Why Big Data Techniques Required
        st.markdown(f"<h3>💡 Why Big Data Techniques Are Required</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        cards = [
            ("📊", "Volume", "8M+ reviews (~3GB raw data) exceeds single-machine memory limits", GOLD),
            ("⚡", "Processing", "Complex aggregations require distributed computing (PySpark)", PURPLE),
            ("🗄️", "Storage", "Columnar formats (Parquet) provide 80% compression for analytics", CORAL)
        ]
        for col, (icon, title, desc, color) in zip([col1, col2, col3], cards):
            with col:
                st.markdown(f"""
                <div style='background: #111; border: 2px solid {color}; border-radius: 10px; padding: 20px; height: 180px;'>
                    <p style='font-size: 2rem; text-align: center;'>{icon}</p>
                    <p style='color: {color}; font-weight: bold; text-align: center;'>{title}</p>
                    <p style='font-size: 0.9rem; text-align: center;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        # Dataset Description
        st.markdown(f"<h3 style='margin-top: 30px;'>📦 Dataset: Stanford SNAP Amazon Movie Reviews</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div style='background: #111; border: 1px solid #333; border-radius: 10px; padding: 20px;'>
                <p style='line-height: 1.8;'>
                    The <strong>Stanford SNAP Amazon Movie Reviews</strong> dataset contains 
                    <strong>7,911,684 movie reviews</strong> spanning <strong>August 1997 to October 2012</strong>. 
                    This is one of the largest publicly available review datasets, making it ideal for 
                    demonstrating cloud-based big data processing techniques.
                </p>
                <h4 style='margin-top: 20px;'>Dataset Schema:</h4>
                <ul>
                    <li><strong>product/productId:</strong> Amazon ASIN identifier</li>
                    <li><strong>review/userId:</strong> Unique reviewer identifier</li>
                    <li><strong>review/score:</strong> Rating (1-5 stars)</li>
                    <li><strong>review/helpfulness:</strong> Helpful votes (e.g., "9/12")</li>
                    <li><strong>review/time:</strong> Unix timestamp</li>
                    <li><strong>review/text:</strong> Full review content</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: #111; border: 2px solid {GOLD}; border-radius: 10px; padding: 20px;'>
                <h4 style='text-align: center;'>📊 Key Statistics</h4>
                <table style='width: 100%;'>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px;'>Total Reviews</td>
                        <td style='padding: 10px; text-align: right;'><strong>7,911,684</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px;'>Unique Users</td>
                        <td style='padding: 10px; text-align: right;'><strong>889,176</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px;'>Unique Products</td>
                        <td style='padding: 10px; text-align: right;'><strong>253,059</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #333;'>
                        <td style='padding: 10px;'>Time Span</td>
                        <td style='padding: 10px; text-align: right;'><strong>15 years</strong></td>
                    </tr>
                    <tr>
                        <td style='padding: 10px;'>Raw Size</td>
                        <td style='padding: 10px; text-align: right;'><strong>~3 GB</strong></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Connection to Real-World Problem
        st.markdown(f"""
        <div style='background: #111; border-left: 4px solid {PURPLE}; padding: 20px; margin: 30px 0;'>
            <h4 style='color: {PURPLE} !important;'>🔗 Connection to the Real-World Problem</h4>
            <p style='line-height: 1.8;'>
                This dataset directly mirrors challenges faced by Netflix, Amazon, and Disney+:
            </p>
            <ul style='line-height: 2;'>
                <li>Processing millions of reviews requires <strong>distributed computing</strong> (AWS Glue/PySpark)</li>
                <li>Efficient storage needs <strong>columnar formats</strong> (Parquet) and cloud storage (S3)</li>
                <li>Complex aggregations require <strong>SQL at scale</strong> (Spark SQL)</li>
                <li>Real-time dashboards need <strong>optimized databases</strong> (MongoDB)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Research Questions
        st.markdown(f"<h3>🔍 Research Questions</h3>", unsafe_allow_html=True)
        
        questions = [
            ("Q1", "How have review volumes and average ratings evolved over the 15-year period?"),
            ("Q2", "What is the distribution of ratings? Is there evidence of selection bias?"),
            ("Q3", "Which types of reviews do users find most helpful?"),
            ("Q4", "How can users be segmented based on their review activity?"),
            ("Q5", "Which products receive the most engagement?")
        ]
        
        for q, question in questions:
            st.markdown(f"""
            <div style='display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid #333;'>
                <span style='background: {ORANGE}; color: #000; padding: 5px 12px; border-radius: 5px; 
                            font-weight: bold; margin-right: 15px;'>{q}</span>
                <span>{question}</span>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 2: METHODOLOGY
    # Big Data techniques, pipeline, tools
    # -----------------------------------------------------------------
    with tab2:
        st.markdown(f"<h2>🔧 Methodology: Cloud & Big Data Pipeline</h2>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background: #111; border: 1px solid {GOLD}; border-radius: 10px; padding: 20px; margin: 20px 0;'>
            <p style='font-size: 1.05rem; text-align: center;'>
                This project implements an <strong>end-to-end cloud-based big data pipeline</strong> using 
                AWS services and techniques taught in ALY6110. <strong>No machine learning</strong> is used—all 
                analysis is performed using <strong>Spark SQL aggregations</strong> on distributed data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Architecture Diagram
        st.markdown(f"<h3>🗺️ Pipeline Architecture</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Data flows from top to bottom through each processing layer.</p>", unsafe_allow_html=True)

        render_pipeline_diagram()  # Use this instead of render_mermaid_diagram()

        # Technology Stack
        st.markdown(f"<h3 style='margin-top: 30px;'>🛠️ Technology Stack (ALY6110 Techniques)</h3>",
                    unsafe_allow_html=True)

        tech_html = f"""
        <table style='width: 100%; border-collapse: collapse; background: #111; border-radius: 10px; overflow: hidden;'>
            <thead>
                <tr style='background: #1a1a1a; border-bottom: 2px solid {GOLD};'>
                    <th style='padding: 12px 15px; text-align: left; color: {GOLD}; width: 18%;'>Technology</th>
                    <th style='padding: 12px 15px; text-align: left; color: {GOLD}; width: 15%;'>Category</th>
                    <th style='padding: 12px 15px; text-align: left; color: {GOLD}; width: 67%;'>Purpose</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>AWS S3 + Boto3</strong></td>
                    <td style='padding: 12px 15px;'>Cloud Storage</td>
                    <td style='padding: 12px 15px;'>Scalable object storage as data lake foundation; Boto3 used to programmatically create bucket with organized folder structure (raw/, staged/, processed/, analytics/)</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>AWS Glue</strong></td>
                    <td style='padding: 12px 15px;'>ETL Service</td>
                    <td style='padding: 12px 15px;'>Serverless ETL orchestration eliminating cluster management; auto-scales infrastructure and integrates natively with S3 and Glue Catalog</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>PySpark</strong></td>
                    <td style='padding: 12px 15px;'>Processing</td>
                    <td style='padding: 12px 15px;'>Distributed engine partitioning 8M reviews across worker nodes for parallel computation; enabled transformations impossible on single machine</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>Parquet</strong></td>
                    <td style='padding: 12px 15px;'>File Format</td>
                    <td style='padding: 12px 15px;'>Columnar format achieving ~80% compression; optimized for analytics by reading only needed columns for each aggregation</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>Spark SQL</strong></td>
                    <td style='padding: 12px 15px;'>Query Engine</td>
                    <td style='padding: 12px 15px;'>Executed complex analytical queries (GROUP BY, aggregations) on distributed data using SQL syntax across Spark cluster</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>Glue Catalog</strong></td>
                    <td style='padding: 12px 15px;'>Data Catalog</td>
                    <td style='padding: 12px 15px;'>Centralized metadata repository storing schemas and partition info; ensures data consistency and enables table discovery</td>
                </tr>
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>MongoDB Atlas</strong></td>
                    <td style='padding: 12px 15px;'>Database</td>
                    <td style='padding: 12px 15px;'>Cloud NoSQL database serving dashboard's data layer; stores pre-computed results enabling sub-second query responses</td>
                </tr>
                <tr>
                    <td style='padding: 12px 15px; color: {ORANGE};'><strong>Streamlit</strong></td>
                    <td style='padding: 12px 15px;'>Dashboard</td>
                    <td style='padding: 12px 15px;'>Python web framework connecting to MongoDB for pre-aggregated results; renders dynamic Plotly visualizations with minimal overhead</td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(tech_html, unsafe_allow_html=True)


        # Implementation Steps
        st.markdown(f"<h3 style='margin-top: 30px;'>📋 Implementation Steps</h3>", unsafe_allow_html=True)

        steps = [
            ("1", "⚙️", "Environment Setup", "Configure AWS credentials and MongoDB connection securely via environment variables.", PURPLE, CODE_STEP1),
            ("2", "☁️", "AWS S3 Infrastructure (Boto3)", "Programmatically create S3 bucket with data lake folder structure (raw/, staged/, processed/, analytics/).", ORANGE, CODE_STEP2),
            ("3", "📥", "Upload Raw Data to S3", "Download 3GB dataset from Stanford SNAP and upload to S3 raw/ prefix for processing.", ORANGE, CODE_STEP3),
            ("4", "⚡", "AWS Glue ETL Job (PySpark)", "Create serverless Glue job with 5 G.1X workers for distributed processing of 8M records.", PURPLE, CODE_STEP4),
            ("5", "🔥", "Spark SQL Analytics", "Execute aggregation queries (GROUP BY, COUNT, AVG) on distributed data—no ML, pure SQL.", CORAL, CODE_SPARK_SQL),
            ("6", "🔍", "Verify Glue Catalog", "Confirm tables registered in Glue Catalog for schema management.", PURPLE, "aws glue get-tables --database-name amazon_movies_db"),
            ("7", "🍃", "Load to MongoDB", "Load pre-computed Spark SQL results to MongoDB for fast dashboard queries.", GOLD, CODE_STEP7),
            ("8", "🎬", "Streamlit Dashboard", "Connect to MongoDB and visualize aggregated results with Plotly.", GOLD, "streamlit run amazon-movie-streamlit-dashboard.py"),
        ]

        for num, icon, title, desc, color, code in steps:
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, {color}15 0%, #0a0a0a 100%); 
                        border-left: 4px solid {color}; padding: 15px 20px; margin: 10px 0;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>{icon}</span>
                <strong style='color: {color} !important;'>Step {num}: {title}</strong>
                <p style='margin: 8px 0 0 35px;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"📄 View Step {num} Code"):
                st.code(code, language="python" if "def " in code or "import " in code or "=" in code else "bash")

    # -----------------------------------------------------------------
    # TAB 3: RESULTS & VISUALIZATIONS (35 pts)
    # Charts, tables, findings
    # -----------------------------------------------------------------
    with tab3:
        st.markdown(f"<h2>📊 Results & Data Visualizations</h2>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: #111; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
            <p style='margin: 0;'>All visualizations below are generated from <strong>pre-computed Spark SQL aggregations</strong> 
            stored in MongoDB. No real-time computation occurs in the dashboard—this demonstrates the 
            <strong>separation of batch processing (Glue) and serving layer (MongoDB)</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        # Chart colors using our 5-color palette
        chart_colors = [CORAL, ORANGE, GOLD, PURPLE, WHITE]

        # Q1: Yearly Trends
        st.markdown(f"<h3>📈 Q1: Review Volume & Ratings Over Time</h3>", unsafe_allow_html=True)

        yearly = stats['yearly_stats']
        if yearly:
            years = [y['_id'] for y in yearly if y['_id'] and y['_id'] >= 1997]
            counts = [y.get('review_count', 0) for y in yearly if y['_id'] and y['_id'] >= 1997]
            avg_ratings = [y.get('avg_rating', 0) for y in yearly if y['_id'] and y['_id'] >= 1997]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=years, y=counts, name="Review Count", marker=dict(color=GOLD), opacity=0.8), secondary_y=False)
            fig.add_trace(go.Scatter(x=years, y=avg_ratings, name="Avg Rating", line=dict(color=CORAL, width=3), mode='lines+markers'), secondary_y=True)
            fig.update_layout(
                title="Review Volume & Average Rating by Year (1997-2012)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0a0a0a', 
                height=400, font=dict(color=WHITE),
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig.update_xaxes(gridcolor='#2a2a2a')
            fig.update_yaxes(title_text="Review Count", secondary_y=False, gridcolor='#2a2a2a')
            fig.update_yaxes(title_text="Avg Rating", secondary_y=True, range=[3.5, 5], gridcolor='#2a2a2a')
            st.plotly_chart(fig, use_container_width=True)

            # Data Table
            yearly_df = pd.DataFrame({'Year': years[-5:], 'Reviews': [f"{c:,}" for c in counts[-5:]], 'Avg Rating': [f"{r:.2f}" for r in avg_ratings[-5:]]})
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div style='background: #111; border-left: 4px solid {GOLD}; padding: 15px;'>
                    <strong>Finding:</strong> Review volume grew exponentially from ~1,000 (1997) to ~2 million (2012), 
                    representing <strong>2000x growth</strong>. Average ratings remained stable at 4.0-4.2★, indicating 
                    consistent user satisfaction despite massive platform growth.
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)

        # Q2: Rating Distribution
        # Q2: Rating Distribution
        st.markdown(f"<h3 style='margin-top: 30px;'>⭐ Q2: Rating Distribution Analysis</h3>", unsafe_allow_html=True)

        rating_dist = stats['rating_distribution']
        if rating_dist:
            scores = [int(r['_id']) for r in rating_dist if r['_id'] and r['_id'] > 0]
            r_counts = [r.get('count', 0) for r in rating_dist if r['_id'] and r['_id'] > 0]

            # Custom color mapping for each star rating
            BLUE = "#3498db"  # Add blue to your colors
            rating_color_map = {
                1: CORAL,  # 1★ - Coral (negative)
                2: BLUE,  # 2★ - Blue (as requested)
                3: GOLD,  # 3★ - Gold (neutral)
                4: PURPLE,  # 4★ - Purple (positive)
                5: ORANGE  # 5★ - Orange (as requested)
            }

            # Create color list based on scores order
            rating_colors = [rating_color_map.get(s, WHITE) for s in scores]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(
                    data=[go.Bar(x=[f"{s}★" for s in scores], y=r_counts, marker=dict(color=rating_colors))])
                fig.update_layout(title="Rating Distribution (Count)", paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='#0a0a0a', height=350, font=dict(color=WHITE))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(data=[
                    go.Pie(labels=[f"{s}★" for s in scores], values=r_counts, marker=dict(colors=rating_colors),
                           hole=0.4)])
                fig.update_layout(title="Rating Distribution (%)", paper_bgcolor='rgba(0,0,0,0)', height=350,
                                  font=dict(color=WHITE))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style='background: #111; border-left: 4px solid {ORANGE}; padding: 15px;'>
                <strong>Finding:</strong> J-shaped distribution with <strong>~60% 5★ ratings</strong> indicates strong 
                <strong>selection bias</strong>—users are more motivated to review when highly satisfied. 
                The secondary peak at 1★ shows polarized opinions are more likely to be expressed.
            </div>
            """, unsafe_allow_html=True)

        # Q3: Helpfulness
        st.markdown(f"<h3 style='margin-top: 30px;'>👍 Q3: Helpfulness by Rating</h3>", unsafe_allow_html=True)

        helpfulness = stats['helpfulness_stats']
        if helpfulness:
            h_scores = [int(h['_id']) for h in helpfulness if h['_id'] and h['_id'] > 0]
            h_ratios = [h.get('avg_helpful_ratio', 0) or 0 for h in helpfulness if h['_id'] and h['_id'] > 0]

            # Use same color mapping as Q2
            helpfulness_colors = [rating_color_map.get(s, WHITE) for s in h_scores]

            fig = go.Figure(data=[go.Bar(
                x=[f"{s}★" for s in h_scores],
                y=h_ratios,
                marker=dict(color=helpfulness_colors),  # Changed from chart_colors
                text=[f'{h:.3f}' for h in h_ratios],
                textposition='outside'
            )])
            fig.update_layout(title="Average Helpfulness Ratio by Rating", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='#0a0a0a', height=400, font=dict(color=WHITE))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style='background: #111; border-left: 4px solid {PURPLE}; padding: 15px;'>
                <strong>Finding:</strong> Critical reviews (1-2★) have <strong>~30% higher helpfulness ratios</strong> 
                than positive reviews. Users actively seek detailed negative feedback before making purchase decisions.
            </div>
            """, unsafe_allow_html=True)

        # Q4: User Segments
        st.markdown(f"<h3 style='margin-top: 30px;'>👥 Q4: User Segmentation</h3>", unsafe_allow_html=True)

        segments = stats['user_segments']
        if segments:
            labels = [s.get('user_segment', s.get('_id', 'Unknown')) for s in segments]
            values = [s.get('num_users', 0) for s in segments]
            reviews = [s.get('total_reviews', 0) for s in segments]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=[CORAL, GOLD, PURPLE]), hole=0.4)])
                fig.update_layout(title="User Segments by Count", paper_bgcolor='rgba(0,0,0,0)', height=350, font=dict(color=WHITE))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(data=[go.Bar(x=labels, y=reviews, marker=dict(color=[CORAL, GOLD, PURPLE]))])
                fig.update_layout(title="Total Reviews by Segment", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0a0a0a', height=350, font=dict(color=WHITE))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style='background: #111; border-left: 4px solid {CORAL}; padding: 15px;'>
                <strong>Finding:</strong> <strong>Power law distribution</strong>: ~2% of users (Power segment with >50 reviews) 
                generate a disproportionate share of content. These ~16,000 power users are critical to the review ecosystem.
            </div>
            """, unsafe_allow_html=True)

        # Q5: Top Products
        st.markdown(f"<h3 style='margin-top: 30px;'>🎥 Q5: Product Engagement</h3>", unsafe_allow_html=True)

        products = stats['product_stats']
        if products:
            p_ids = [p['_id'][:12] + '...' if len(str(p['_id'])) > 12 else p['_id'] for p in products[:10]]
            p_counts = [p.get('review_count', 0) for p in products[:10]]
            p_ratings = [p.get('avg_rating', 0) or 4.0 for p in products[:10]]

            fig = go.Figure(data=[go.Bar(y=p_ids, x=p_counts, orientation='h',
                                         marker=dict(color=p_ratings, colorscale=[[0, CORAL], [0.5, ORANGE], [1, GOLD]],
                                                     colorbar=dict(title='Avg Rating')))])
            fig.update_layout(title="Top 10 Most Reviewed Products", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='#0a0a0a', height=400, font=dict(color=WHITE),
                              yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style='background: #111; border-left: 4px solid {GOLD}; padding: 15px;'>
                <strong>Finding:</strong> <strong>Blockbuster effect</strong>: Top products receive 100x more reviews than median. 
                High review count doesn't guarantee highest ratings—popularity and quality are distinct metrics.
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 4: CONCLUSIONS & RECOMMENDATIONS (25 pts)
    # -----------------------------------------------------------------
    with tab4:
        st.markdown(f"<h2>📋 Conclusions & Recommendations</h2>", unsafe_allow_html=True)

        # Summary answering the research questions
        st.markdown(f"""
        <div style='background: #111; border: 2px solid {GOLD}; border-radius: 15px; padding: 25px; margin: 20px 0;'>
            <h3 style='text-align: center;'>🎯 Answering the Research Questions</h3>
            <p style='text-align: center; margin-top: 15px;'>
                Using <strong>Spark SQL aggregations</strong> on 8 million reviews processed via AWS Glue, 
                we derived the following insights:
            </p>
        </div>
        """, unsafe_allow_html=True)

        conclusions = [
            ("Q1", "Temporal Trends", "Review volume grew 2000x (1997-2012) while average ratings remained stable at ~4.1★, indicating consistent user satisfaction despite exponential platform growth.", GOLD),
            ("Q2", "Rating Distribution", "J-shaped distribution with 60% 5★ ratings reveals significant selection bias—satisfied customers are more likely to leave reviews.", ORANGE),
            ("Q3", "Helpfulness", "Critical reviews (1-2★) are 30% more helpful than positive ones. Users seek detailed negative feedback for informed decision-making.", PURPLE),
            ("Q4", "User Segments", "Power law applies: 2% of users (Power reviewers with >50 reviews) drive ecosystem health. These ~16,000 users are critical stakeholders.", CORAL),
            ("Q5", "Product Engagement", "Blockbuster effect: Top 1% of products receive ~50% of reviews. Popularity ≠ quality.", GOLD)
        ]

        for q, title, finding, color in conclusions:
            st.markdown(f"""
            <div style='background: #111; border-left: 4px solid {color}; padding: 15px; margin: 12px 0;'>
                <span style='background: {color}; color: #000; padding: 3px 10px; border-radius: 5px; font-weight: bold;'>{q}</span>
                <strong style='margin-left: 10px;'>{title}</strong>
                <p style='margin: 10px 0 0 0;'>{finding}</p>
            </div>
            """, unsafe_allow_html=True)

        # Recommendations
        st.markdown(f"<h3 style='margin-top: 30px;'>💡 Actionable Recommendations</h3>", unsafe_allow_html=True)

        recommendations = [
            ("1", "🎯", "Surface Critical Reviews", "Since negative reviews are 30% more helpful, platforms should feature them prominently alongside positive reviews to help customers make informed decisions.", "Increase customer trust"),
            ("2", "👥", "Power User Programs", "The ~16,000 power users who write >50 reviews are critical to ecosystem health. Implement loyalty programs, early access, or recognition systems to retain them.", "Maintain review volume"),
            ("3", "📊", "Address Selection Bias", "With 60% 5★ ratings, prompt neutral/dissatisfied customers to review via follow-up emails targeting non-reviewers.", "Representative feedback"),
            ("4", "🔍", "Separate Popularity from Quality", "High review count ≠ high quality. Create distinct 'Most Popular' and 'Highest Rated' categories in product discovery.", "Improve recommendations"),
            ("5", "📈", "Monitor Rating Trends", "Stable ratings despite growth is healthy. Any sustained rating decline should trigger immediate product quality investigation.", "Proactive management")
        ]

        for num, icon, title, desc, impact in recommendations:
            st.markdown(f"""
            <div style='background: #111; border: 1px solid #333; border-radius: 10px; padding: 20px; margin: 15px 0;'>
                <span style='font-size: 1.8rem; margin-right: 15px;'>{icon}</span>
                <strong style='color: {GOLD} !important;'>Recommendation {num}: {title}</strong>
                <p style='margin: 10px 0 5px 45px;'>{desc}</p>
                <p style='margin: 0 0 0 45px; color: {ORANGE} !important;'><strong>Expected Impact:</strong> {impact}</p>
            </div>
            """, unsafe_allow_html=True)

        # Technical Achievement
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {GOLD}15 0%, {PURPLE}15 100%); 
                    border: 2px solid {GOLD}; border-radius: 15px; padding: 25px; margin: 30px 0;'>
            <h3 style='text-align: center;'>🏆 Technical Achievement</h3>
            <p style='text-align: center; font-size: 1.05rem; margin-top: 15px;'>
                Successfully implemented a <strong>cloud-native big data pipeline</strong> that processed 
                <strong>8 million records</strong> using:
            </p>
            <p style='text-align: center; margin-top: 15px;'>
                <span style='background: {ORANGE}; color: #000; padding: 5px 15px; border-radius: 5px; margin: 5px;'>AWS S3 + Boto3</span>
                <span style='background: {PURPLE}; color: #fff; padding: 5px 15px; border-radius: 5px; margin: 5px;'>AWS Glue (PySpark)</span>
                <span style='background: {CORAL}; color: #fff; padding: 5px 15px; border-radius: 5px; margin: 5px;'>Spark SQL</span>
                <span style='background: {GOLD}; color: #000; padding: 5px 15px; border-radius: 5px; margin: 5px;'>MongoDB Atlas</span>
            </p>
            <p style='text-align: center; margin-top: 15px;'>
                All analysis performed using <strong>SQL aggregations</strong>—no machine learning techniques were used, 
                demonstrating practical application of <strong>ALY6110 Cloud & Big Data Management</strong> concepts.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Limitations
        st.markdown(f"<h3 style='margin-top: 30px;'>⚠️ Limitations</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background: #111; border: 1px solid #333; border-radius: 10px; padding: 20px;'>
            <ul style='line-height: 2;'>
                <li><strong>Data Age:</strong> Dataset ends in 2012; modern streaming behavior patterns not captured</li>
                <li><strong>Sampling:</strong> MongoDB free tier (512MB) required 150K sample for dashboard serving layer</li>
                <li><strong>No Product Metadata:</strong> Product titles/categories unavailable for deeper product analysis</li>
                <li><strong>Selection Bias:</strong> Only voluntary reviewers represented—silent majority excluded</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style='text-align: center; padding: 25px 0; margin-top: 30px; border-top: 2px solid {GOLD};'>
        <p style='color: {GOLD}; font-size: 1.1rem;'>ALY6110 - Cloud & Big Data Management | Final Project</p>
        <p style='font-size: 0.9rem;'>AWS S3 • Boto3 • AWS Glue • PySpark • Spark SQL • Parquet • MongoDB • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
