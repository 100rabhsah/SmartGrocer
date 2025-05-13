# app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apriori_analysis import run_apriori
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="SmartGrocer - Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🛒 SmartGrocer - Market Basket Analysis")
st.markdown("Discover shopping patterns and product relationships in your store's transaction data.")

# Educational Section
with st.expander("📚 How Market Basket Analysis Works"):
    st.markdown("""
    ### Understanding Market Basket Analysis
    
    Market Basket Analysis helps you understand what products customers buy together. Here's a simple example:
    
    **Example Scenario:**
    - 100 customers in your store
    - 10 customers bought milk
    - 8 customers bought butter
    - 6 customers bought both milk and butter
    
    **The Rule:** If a customer buys milk, they are likely to buy butter
    
    **Key Metrics:**
    1. **Support** (How common is this combination?)
       - Support = (Number of transactions with both items) / (Total transactions)
       - In our example: 6/100 = 0.06 or 6%
    
    2. **Confidence** (How reliable is this rule?)
       - Confidence = (Number of transactions with both items) / (Number of transactions with first item)
       - In our example: 6/8 = 0.75 or 75%
    
    3. **Lift** (How much more likely is this combination?)
       - Lift = Confidence / (Probability of second item)
       - In our example: 0.75/0.10 = 7.5
    
    **What do these numbers mean?**
    - Higher Support: More common combination
    - Higher Confidence: More reliable rule
    - Lift > 1: Positive correlation between items
    - Lift = 1: No correlation
    - Lift < 1: Negative correlation
    
    **Note:** In real-world scenarios, you need hundreds of transactions for meaningful patterns.
    """)

# Data Selection Section
st.subheader("📂 Choose Your Data Source")
data_source = st.radio(
    "Select how you want to provide your data:",
    ["Upload your own CSV file", "Use sample dataset", "Use Groceries dataset"],
    horizontal=True
)

data = None

if data_source == "Upload your own CSV file":
    st.markdown("""
    Upload your transaction data in CSV format. The file should have these columns:
    - Member_number: Customer ID
    - Date: Transaction date
    - itemDescription: Product name
    """)
    
    uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.info("Please make sure your CSV file has the correct format and columns.")
elif data_source == "Use sample dataset":
    # Use sample dataset
    sample_file = os.path.join("data", "sample_transactions.csv")
    if os.path.exists(sample_file):
        try:
            data = pd.read_csv(sample_file)
            st.success("✅ Successfully loaded sample dataset!")
            st.info("This is a small sample dataset with grocery store transactions. You can use this to explore the app's features.")
        except Exception as e:
            st.error(f"Error reading the sample file: {str(e)}")
    else:
        st.error("Sample dataset not found. Please upload your own CSV file or contact support.")
else:  # Use Groceries dataset
    groceries_file = os.path.join("data", "Groceries_dataset.csv")
    if os.path.exists(groceries_file):
        try:
            data = pd.read_csv(groceries_file)
            st.success("✅ Successfully loaded Groceries dataset!")
            st.info("""
            This is a comprehensive dataset of grocery store transactions. It includes:
            - Real-world shopping patterns
            - Multiple product categories
            - Extended time period
            - Large number of transactions
            """)
        except Exception as e:
            st.error(f"Error reading the Groceries dataset: {str(e)}")
    else:
        st.error("Groceries dataset not found. Please upload your own CSV file or use the sample dataset.")

if data is not None:
    try:
        # Data Preview Section
        st.subheader("📊 Data Preview")
        st.markdown("Here's a preview of your data. Make sure the format is correct.")
        st.dataframe(data.head(), use_container_width=True)
        
        # Basic Statistics Section
        st.subheader("📈 Store Overview")
        
        # Calculate dynamic statistics
        total_customers = len(data['Member_number'].unique())
        total_products = len(data['itemDescription'].unique())
        date_range = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        start_date = date_range.min().strftime('%B %Y')
        end_date = date_range.max().strftime('%B %Y')
        
        # Calculate average purchases per customer
        purchases_per_customer = data.groupby('Member_number').size()
        avg_purchases = purchases_per_customer.mean()
        
        # Simple Summary Box with dynamic statistics
        st.markdown(f"""
        <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(128, 128, 128, 0.2);'>
            <h3 style='color: #1f77b4; margin-bottom: 15px;'>📊 Quick Summary</h3>
            <p style='font-size: 18px; color: #2c3e50;'>Your store has:</p>
            <ul style='font-size: 16px; color: #2c3e50; list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 8px;'>👥 <strong>{total_customers:,} loyal customers</strong></li>
                <li style='margin-bottom: 8px;'>🛍️ <strong>{total_products:,} different products</strong> on your shelves</li>
                <li style='margin-bottom: 8px;'>📅 Data from <strong>{start_date} to {end_date}</strong></li>
            </ul>
            <p style='font-size: 18px; margin-top: 15px; color: #2c3e50;'>💡 <strong>Key Insight:</strong> Each customer shops about {avg_purchases:.1f} times in your store</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visual Dashboard
        st.markdown("### 📊 Store Performance Dashboard")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #e6f3ff; border-radius: 10px;'>
                <h3 style='color: #1f77b4;'>👥 Customers</h3>
                <h2 style='font-size: 36px; color: #1f77b4;'>{:,}</h2>
                <p style='color: #666;'>Total Loyal Customers</p>
            </div>
            """.format(len(data['Member_number'].unique())), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #e6ffe6; border-radius: 10px;'>
                <h3 style='color: #2e7d32;'>🛍️ Products</h3>
                <h2 style='font-size: 36px; color: #2e7d32;'>{:,}</h2>
                <p style='color: #666;'>Different Items in Store</p>
            </div>
            """.format(len(data['itemDescription'].unique())), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #fff3e6; border-radius: 10px;'>
                <h3 style='color: #e65100;'>📅 Period</h3>
                <h2 style='font-size: 24px; color: #e65100;'>{}</h2>
                <p style='color: #666;'>Data Collection Period</p>
            </div>
            """.format(f"{data['Date'].min()} to {data['Date'].max()}"), unsafe_allow_html=True)
        
        # Customer Shopping Patterns
        st.markdown("### 👥 Customer Shopping Patterns")
        
        # Calculate purchases per customer
        purchases_per_customer = data.groupby('Member_number').size()
        
        # Create a more visually appealing histogram
        fig = px.histogram(
            x=purchases_per_customer.values,
            nbins=30,
            labels={'x': 'Number of Purchases', 'y': 'Number of Customers'},
            title='How Often Do Customers Shop?',
            color_discrete_sequence=['#1f77b4']
        )
        
        # Update layout for better readability
        fig.update_layout(
            plot_bgcolor='white',
            xaxis_title="Number of Times a Customer Shops",
            yaxis_title="Number of Customers",
            showlegend=False,
            title_x=0.5,
            title_font_size=20
        )
        
        # Add a vertical line for the average
        fig.add_vline(
            x=purchases_per_customer.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Average",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Simple interpretation
        st.markdown("""
        <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(128, 128, 128, 0.2);'>
            <h4 style='color: #1f77b4; margin-bottom: 15px; font-weight: bold;'>💡 What This Means for Your Store:</h4>
            <ul style='font-size: 16px; color: #2c3e50; list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 8px;'>• Most customers shop between {} and {} times</li>
                <li style='margin-bottom: 8px;'>• The average customer makes about {:.1f} purchases</li>
                <li style='margin-bottom: 8px;'>• Your most loyal customer has made {} purchases</li>
            </ul>
        </div>
        """.format(
            int(purchases_per_customer.quantile(0.25)),
            int(purchases_per_customer.quantile(0.75)),
            purchases_per_customer.mean(),
            int(purchases_per_customer.max())
        ), unsafe_allow_html=True)
        
        # Daily Transaction Pattern
        st.markdown("### 📅 Daily Shopping Patterns")
        
        # Calculate daily transactions
        daily_transactions = data.groupby(data['Date']).size().reset_index()
        daily_transactions.columns = ['Date', 'Transactions']
        
        # Create a line chart for daily transactions
        fig = px.line(
            daily_transactions,
            x='Date',
            y='Transactions',
            title='Daily Shopping Activity',
            labels={'Transactions': 'Number of Items Sold', 'Date': 'Date'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="Number of Items Sold",
            showlegend=False,
            title_x=0.5,
            title_font_size=20,
            font=dict(
                family="Arial",
                size=12,
                color="#2c3e50"
            )
        )
        
        # Update axes for better visibility
        fig.update_xaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color="#2c3e50")
        )
        fig.update_yaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color="#2c3e50")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Simple interpretation of daily patterns with improved visibility
        st.markdown("""
        <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(128, 128, 128, 0.2);'>
            <h4 style='color: #1f77b4; margin-bottom: 15px; font-weight: bold;'>💡 Understanding Your Daily Sales:</h4>
            <ul style='font-size: 16px; color: #2c3e50; list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 8px;'>• On average, you sell about {:.1f} items per day</li>
                <li style='margin-bottom: 8px;'>• This helps you plan your inventory and staffing</li>
                <li style='margin-bottom: 8px;'>• Look for patterns in busy and quiet days</li>
            </ul>
        </div>
        """.format(daily_transactions['Transactions'].mean()), unsafe_allow_html=True)
        
        # Analysis Parameters Section
        st.subheader("⚙️ Analysis Parameters")
        st.markdown("""
        Adjust these parameters to find meaningful patterns:
        - **Minimum Support**: How common should the item combination be? (e.g., 0.1 = 10% of transactions)
        - **Minimum Confidence**: How reliable should the rule be? (e.g., 0.5 = 50% confidence)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 
                                  help="Minimum frequency of itemsets in the dataset")
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5,
                                     help="Minimum probability of itemset Y being purchased when itemset X is purchased")
        
        # Run analysis
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing transaction patterns..."):
                # Run Apriori algorithm
                frequent_itemsets, rules, binary_data, stats = run_apriori(data, min_support, min_confidence)
                
                # Transaction Statistics Section
                st.subheader("📊 Transaction Statistics")
                st.markdown("Key insights about your transaction data")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Items per Transaction", 
                             f"{stats['avg_items_per_transaction']:.2f}")
                with col2:
                    st.metric("Most Common Item", 
                             list(stats['most_common_items'].keys())[0])
                
                # Transaction Patterns Section
                st.subheader("📈 Transaction Patterns Over Time")
                st.markdown("How shopping patterns change over time")
                fig = px.line(x=list(stats['transaction_by_date'].keys()),
                            y=list(stats['transaction_by_date'].values()),
                            labels={'x': 'Date', 'y': 'Number of Transactions'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Association Rules Section
                st.subheader("🔍 Association Rules")
                st.markdown("""
                These rules show which items are frequently bought together.
                - **Antecedents**: The "if" part of the rule
                - **Consequents**: The "then" part of the rule
                - **Support**: How common is this combination?
                - **Confidence**: How reliable is this rule?
                - **Lift**: How much more likely is this combination?
                """)
                
                if not rules.empty:
                    # Format rules for display
                    display_rules = rules.copy()
                    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Display rules with metrics
                    st.dataframe(
                        display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                        use_container_width=True
                    )
                    
                    # Visualize top rules
                    st.subheader("📊 Top Rules by Lift")
                    st.markdown("""
                    This visualization shows the strongest rules:
                    - **X-axis**: Support (how common)
                    - **Y-axis**: Confidence (how reliable)
                    - **Bubble size**: Lift (how much more likely)
                    - **Hover**: See all details
                    """)
                    
                    top_rules = rules.head(10).copy()
                    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Create hover text
                    top_rules['hover_text'] = top_rules.apply(
                        lambda x: f"Rule: {x['antecedents']} → {x['consequents']}<br>" +
                                f"Support: {x['support']:.3f}<br>" +
                                f"Confidence: {x['confidence']:.3f}<br>" +
                                f"Lift: {x['lift']:.3f}",
                        axis=1
                    )
                    
                    # Create scatter plot
                    fig = px.scatter(
                        top_rules,
                        x='support',
                        y='confidence',
                        size='lift',
                        hover_data=['hover_text'],
                        title='Top 10 Rules by Lift'
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Support",
                        yaxis_title="Confidence",
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analysis Results Section
                    st.subheader("🔍 What Products Do Customers Buy Together?")
                    
                    # Simple explanation box
                    st.markdown("""
                    <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>💡 Understanding Product Relationships</h3>
                        <p style='font-size: 16px; color: inherit;'>We found some interesting patterns in how customers shop:</p>
                        <ul style='font-size: 16px; color: inherit;'>
                            <li>When customers buy certain products, they often buy other specific products</li>
                            <li>These patterns can help you with product placement and promotions</li>
                            <li>The stronger the relationship, the more likely customers are to buy these items together</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display top 3 rules in a user-friendly format
                    st.markdown("### 🏆 Top 3 Strongest Product Relationships")
                    
                    for idx, rule in rules.head(3).iterrows():
                        antecedents = ', '.join(list(rule['antecedents']))
                        consequents = ', '.join(list(rule['consequents']))
                        st.markdown(f"""
                        <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                            <h4 style='color: #1f77b4; margin-bottom: 10px;'>Relationship {idx + 1}</h4>
                            <p style='font-size: 16px; color: inherit;'>
                                When customers buy <strong>{antecedents}</strong>,<br>
                                they often also buy <strong>{consequents}</strong>
                            </p>
                            <div style='display: flex; justify-content: space-between; margin-top: 10px;'>
                                <div style='text-align: center; flex: 1;'>
                                    <p style='font-size: 14px; color: inherit;'>How Common</p>
                                    <p style='font-size: 18px; color: #2e7d32;'>{rule['support']:.1%}</p>
                                </div>
                                <div style='text-align: center; flex: 1;'>
                                    <p style='font-size: 14px; color: inherit;'>How Reliable</p>
                                    <p style='font-size: 18px; color: #1f77b4;'>{rule['confidence']:.1%}</p>
                                </div>
                                <div style='text-align: center; flex: 1;'>
                                    <p style='font-size: 14px; color: inherit;'>Strength</p>
                                    <p style='font-size: 18px; color: #e65100;'>{rule['lift']:.1f}x</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualize top rules with improved styling
                    st.markdown("### 📊 Visualizing Product Relationships")
                    st.markdown("""
                    <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <p style='font-size: 16px; color: inherit;'>This chart shows the strongest product relationships:</p>
                        <ul style='font-size: 16px; color: inherit;'>
                            <li>Bigger bubbles = stronger relationships</li>
                            <li>Higher up = more reliable patterns</li>
                            <li>Further right = more common combinations</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create scatter plot with improved styling
                    fig = px.scatter(
                        top_rules,
                        x='support',
                        y='confidence',
                        size='lift',
                        hover_data=['hover_text'],
                        title='Product Relationship Strength'
                    )
                    
                    # Update layout for better visibility in both modes
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="How Common (Support)",
                        yaxis_title="How Reliable (Confidence)",
                        showlegend=False,
                        title_x=0.5,
                        title_font_size=20,
                        font=dict(
                            family="Arial",
                            size=12,
                            color="#2c3e50"  # Dark gray that works in both modes
                        )
                    )
                    
                    # Update axes for better visibility
                    fig.update_xaxes(
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(color="#2c3e50")
                    )
                    fig.update_yaxes(
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(color="#2c3e50")
                    )
                    
                    # Update traces for better visibility
                    fig.update_traces(
                        marker=dict(
                            color='#1f77b4',  # Blue color for points
                            line=dict(
                                color='#ffffff',  # White border
                                width=1
                            )
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actionable insights
                    st.markdown("### 💡 How to Use This Information")
                    st.markdown("""
                    <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <h4 style='color: #1f77b4; margin-bottom: 15px;'>Practical Tips for Your Store:</h4>
                        <ul style='font-size: 16px; color: inherit;'>
                            <li>Place related products near each other to encourage more sales</li>
                            <li>Create special offers for products that are often bought together</li>
                            <li>Use these insights to plan your inventory and promotions</li>
                            <li>Consider creating bundle deals for strongly related products</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("""
                    <div style='background-color: rgba(240, 242, 246, 0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <h4 style='color: #e65100; margin-bottom: 15px;'>No Strong Patterns Found</h4>
                        <p style='font-size: 16px; color: inherit;'>Try these adjustments:</p>
                        <ul style='font-size: 16px; color: inherit;'>
                            <li>Lower the minimum support (how common a pattern should be)</li>
                            <li>Lower the minimum confidence (how reliable a pattern should be)</li>
                            <li>Make sure you have enough transaction data</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export Section
                st.subheader("💾 Export Results")
                st.markdown("Download your analysis results for further use")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export Rules to CSV"):
                        export_rules = rules.copy()
                        export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        export_rules.to_csv("association_rules.csv", index=False)
                        st.success("Rules exported successfully!")
                with col2:
                    if st.button("Export Statistics to CSV"):
                        pd.DataFrame(stats).to_csv("transaction_stats.csv", index=False)
                        st.success("Statistics exported successfully!")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("""
        Please make sure your CSV file has these columns:
        - Member_number: Customer ID
        - Date: Transaction date
        - itemDescription: Product name
        """)
