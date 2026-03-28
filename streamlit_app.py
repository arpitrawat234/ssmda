import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

st.title("📊 Full Stock Analysis Tool (Phase 1-3)")

# -------------------------------
# Upload CSV
uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    stock_name = os.path.basename(uploaded_file.name).split('.')[0].upper()
    st.subheader(f"Stock: {stock_name}")
    
    # -------------------------------
    # Phase 1: EDA
    df['Daily_Return'] = df['Adj Close'].pct_change() * 100
    df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)
    
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Descriptive Statistics")
    st.write(df[['Volume', 'Daily_Return']].describe())
    
    # Plot Adj Close
    st.write("### Adjusted Close Price Over Time")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Date'], df['Adj Close'], color='blue', label='Adj Close')
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.grid(True)
    st.pyplot(fig)
    
    # Daily Return Histogram
    st.write("### Daily Return Distribution")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.hist(df['Daily_Return'].dropna(), bins=50, edgecolor='black')
    ax2.set_xlabel("Return (%)"); ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    
    # Hypothesis Testing: Volume on Up vs Down Days
    df['Up_Day'] = df['Daily_Return'] > 0
    volume_up = df[df['Up_Day']]['Volume']
    volume_down = df[~df['Up_Day']]['Volume']
    t_stat, p_value = stats.ttest_ind(volume_up, volume_down, equal_var=False)
    st.write("### Hypothesis Test: Volume on Up vs Down Days")
    st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}")
    if p_value < 0.05:
        st.write("✅ Significant difference in volume")
    else:
        st.write("❌ No significant difference in volume")
    
    # Boxplot
    st.write("### Volume Comparison Boxplot")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.boxplot([volume_up, volume_down], labels=["Up Days", "Down Days"])
    ax3.set_ylabel("Volume")
    st.pyplot(fig3)
    
    # -------------------------------
    # Phase 2: Feature Optimization via PCA
    st.write("## Phase 2: PCA / Feature Optimization")
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    
    st.write("### Explained Variance Ratio by Principal Components")
    st.bar_chart(explained_var)
    
    st.write("### Principal Components (First 2) Sample")
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    st.dataframe(df_pca[['PC1','PC2']].head())
    
    st.write(f"📌 PC1 explains the most variance: {explained_var[0]:.2f}")
    
    # -------------------------------
    # Phase 3: Regression Models
    st.write("## Phase 3: Predictive Models")
    
    # Prepare data
    df_model = df.dropna(subset=['Open','High','Low','Volume','Target_Next_Day_Close'])
    X_reg = df_model[['Open','High','Low','Volume']]
    y_reg = df_model['Target_Next_Day_Close']
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_reg, y_reg)
    y_pred = lin_reg.predict(X_reg)
    
    st.write("### Linear Regression Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X_reg.columns,
        "Coefficient": lin_reg.coef_
    })
    st.dataframe(coef_df)
    
    # Residuals plot
    residuals = y_reg - y_pred
    st.write("### Residuals Plot")
    fig4, ax4 = plt.subplots(figsize=(10,5))
    ax4.scatter(y_pred, residuals, alpha=0.5)
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Residuals")
    st.pyplot(fig4)
    
    # Logistic Regression: Next day up or down
    df_model['Target_Up'] = (df_model['Target_Next_Day_Close'] > df_model['Adj Close']).astype(int)
    X_log = df_model[['Open','High','Low','Volume']]
    y_log = df_model['Target_Up']
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_log, y_log)
    y_prob = log_reg.predict_proba(X_log)[:,1]
    
    st.write("### Logistic Regression: Next Day Up Probability Sample")
    df_model['Pred_Prob_Up'] = y_prob
    st.dataframe(df_model[['Date','Adj Close','Target_Next_Day_Close','Pred_Prob_Up']].head())
