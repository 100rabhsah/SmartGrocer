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
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🛒 SmartGrocer - Market Basket Analysis")
st.markdown("Analyze shopping patterns and discover product associations in your transaction data.")

# File uploader
uploaded_file = st.file_uploader("Upload your grocery transaction CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and display data preview
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(), use_container_width=True)
        
        # Display basic statistics
        st.subheader("📈 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(data['Member_number'].unique()))
        with col2:
            st.metric("Total Items", len(data['itemDescription'].unique()))
        with col3:
            st.metric("Date Range", f"{data['Date'].min()} to {data['Date'].max()}")
        
        # Parameters
        st.subheader("⚙️ Analysis Parameters")
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
                
                # Display transaction statistics
                st.subheader("📊 Transaction Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Items per Transaction", 
                             f"{stats['avg_items_per_transaction']:.2f}")
                with col2:
                    st.metric("Most Common Item", 
                             list(stats['most_common_items'].keys())[0])
                
                # Visualize transaction patterns over time
                st.subheader("📈 Transaction Patterns Over Time")
                fig = px.line(x=list(stats['transaction_by_date'].keys()),
                            y=list(stats['transaction_by_date'].values()),
                            labels={'x': 'Date', 'y': 'Number of Transactions'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display association rules
                st.subheader("🔍 Association Rules")
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
                    
                    # Visualize top rules by lift
                    st.subheader("📊 Top Rules by Lift")
                    top_rules = rules.head(10).copy()
                    
                    # Convert frozensets to strings for Plotly
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
                    st.warning("No association rules found with the current parameters. Try lowering the minimum support or confidence.")
                
                # Export options
                st.subheader("💾 Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export Rules to CSV"):
                        # Convert frozensets to strings before exporting
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
        st.info("Please make sure your CSV file has the following columns: Member_number, Date, itemDescription")
