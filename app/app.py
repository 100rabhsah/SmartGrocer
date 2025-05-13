# app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apriori_analysis import run_apriori
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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

# File uploader
st.subheader("📂 Upload Your Data")
st.markdown("""
Upload your transaction data in CSV format. The file should have these columns:
- Member_number: Customer ID
- Date: Transaction date
- itemDescription: Product name
""")

uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and display data preview
        data = pd.read_csv(uploaded_file)
        
        # Data Preview Section
        st.subheader("📊 Data Preview")
        st.markdown("Here's a preview of your data. Make sure the format is correct.")
        st.dataframe(data.head(), use_container_width=True)
        
        # Basic Statistics Section
        st.subheader("📈 Basic Statistics")
        st.markdown("""
        ### Understanding Your Data
        
        Your dataset contains:
        - **38,765 individual purchases** (rows in the dataset)
        - **3,898 unique customers** (Member_number)
        - **167 different products** (itemDescription)
        
        This means:
        - On average, each customer made about 10 purchases (38,765 ÷ 3,898 ≈ 10)
        - The data spans from January 1, 2014 to October 31, 2015
        """)
        
        # Calculate additional statistics
        total_purchases = len(data)
        avg_purchases_per_customer = total_purchases / len(data['Member_number'].unique())
        total_days = (pd.to_datetime(data['Date'].max()) - pd.to_datetime(data['Date'].min())).days
        avg_daily_transactions = total_purchases / total_days
        
        # Display detailed statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Individual Purchases", f"{total_purchases:,}")
            st.metric("Unique Customers", f"{len(data['Member_number'].unique()):,}")
            st.metric("Different Products", f"{len(data['itemDescription'].unique()):,}")
        with col2:
            st.metric("Average Purchases per Customer", f"{avg_purchases_per_customer:.1f}")
            st.metric("Average Daily Transactions", f"{avg_daily_transactions:.1f}")
            st.metric("Date Range", f"{data['Date'].min()} to {data['Date'].max()}")
        
        # Add explanation of the numbers
        st.markdown("""
        ### What These Numbers Mean
        
        1. **Total Individual Purchases (38,765)**
           - This is the total number of items purchased
           - Each row represents one item bought by a customer
           - For example, if a customer buys milk, bread, and eggs in one visit, that's 3 purchases
        
        2. **Unique Customers (3,898)**
           - This is the number of different customers who made purchases
           - Each customer is identified by their Member_number
           - On average, each customer made about 10 purchases
        
        3. **Different Products (167)**
           - This is the number of unique items available in the store
           - Each product is counted only once, regardless of how many times it was purchased
        
        4. **Average Daily Transactions**
           - This shows how many items are typically purchased per day
           - Helps understand the store's daily activity level
        """)
        
        # Add a visualization of purchase distribution
        st.subheader("📊 Purchase Distribution")
        st.markdown("How purchases are distributed across customers")
        
        # Calculate purchases per customer
        purchases_per_customer = data.groupby('Member_number').size()
        
        # Create histogram of purchases per customer
        fig = px.histogram(
            x=purchases_per_customer.values,
            nbins=50,
            labels={'x': 'Number of Purchases', 'y': 'Number of Customers'},
            title='Distribution of Purchases per Customer'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary of purchase distribution
        st.markdown(f"""
        **Purchase Distribution Summary:**
        - Most customers made between {purchases_per_customer.quantile(0.25):.0f} and {purchases_per_customer.quantile(0.75):.0f} purchases
        - The median number of purchases per customer is {purchases_per_customer.median():.0f}
        - The maximum number of purchases by a single customer is {purchases_per_customer.max():.0f}
        """)
        
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
                else:
                    st.warning("""
                    No association rules found with the current parameters. Try:
                    1. Lowering the minimum support
                    2. Lowering the minimum confidence
                    3. Checking if your data has enough transactions
                    """)
                
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
