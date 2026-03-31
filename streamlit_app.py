import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MAIT DA-304T | Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Header banner */
  .mait-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c81 100%);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #f59e0b;
  }
  .mait-header h1 { color: #f8fafc; font-size: 1.6rem; margin: 0; font-weight: 700; }
  .mait-header p  { color: #94a3b8; font-size: 0.82rem; margin: 4px 0 0; font-family: 'IBM Plex Mono', monospace; }

  /* Phase cards */
  .phase-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .phase-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    border-bottom: 2px solid #f59e0b;
    padding-bottom: 6px;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
  }

  /* Metric boxes */
  .metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
  .metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    flex: 1; min-width: 130px;
  }
  .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 1.3rem; font-weight: 700; color: #f59e0b; font-family: 'IBM Plex Mono', monospace; }
  .metric-sub   { font-size: 0.72rem; color: #94a3b8; }

  /* Hypothesis box */
  .hyp-box {
    background: #0c1220;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #cbd5e1;
    margin-bottom: 1rem;
    line-height: 1.8;
  }
  .hyp-box strong { color: #f59e0b; }

  /* Info / warning overrides */
  div[data-testid="stInfo"]    { background: #0c1e38; border-color: #1e5fa0; color: #93c5fd; }
  div[data-testid="stSuccess"] { background: #052e16; border-color: #166534; }
  div[data-testid="stWarning"] { background: #1c1302; border-color: #854d0e; }

  /* Table */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0b1424; }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Hide GitHub icon */
  .css-1jc7ptx, .css-1rs6yo7,
  [data-testid="stToolbar"] { display: none !important; }
  header[data-testid="stHeader"] a[href*="github"] { display: none !important; }

  /* Code / mono spans */
  code { background: #1e293b; border-radius: 4px; padding: 2px 6px;
         font-family: 'IBM Plex Mono', monospace; font-size: 0.8em; color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0f172a',
    'axes.facecolor':    '#1e293b',
    'axes.edgecolor':    '#334155',
    'axes.labelcolor':   '#cbd5e1',
    'xtick.color':       '#64748b',
    'ytick.color':       '#64748b',
    'grid.color':        '#334155',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'text.color':        '#cbd5e1',
    'legend.facecolor':  '#1e293b',
    'legend.edgecolor':  '#334155',
    'font.family':       'monospace',
})
AMBER   = '#f59e0b'
SKY     = '#38bdf8'
ROSE    = '#fb7185'
EMERALD = '#34d399'
SLATE   = '#94a3b8'

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="mait-header">
  <h1>📊 Stock Market Analysis & Prediction Tool</h1>
  <p>MAHARAJA AGRASEN INSTITUTE OF TECHNOLOGY &nbsp;|&nbsp; DA-304T Statistics, Statistical Modelling & Data Analytics
     &nbsp;|&nbsp; Semester 6 &nbsp;|&nbsp; AY 2025-26</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Pre-loaded stocks from GitHub repo
# ─────────────────────────────────────────────
import io, requests

PRELOADED_STOCKS = {
    "AAPL – Apple Inc.":       "https://raw.githubusercontent.com/arpitrawat234/ssmda/main/AAPL.csv",
    "AMZN – Amazon.com Inc.":  "https://raw.githubusercontent.com/arpitrawat234/ssmda/main/AMZN.csv",
    "GOOGL – Alphabet Inc.":   "https://raw.githubusercontent.com/arpitrawat234/ssmda/main/GOOGL.csv",
    "MSFT – Microsoft Corp.":  "https://raw.githubusercontent.com/arpitrawat234/ssmda/main/MSFT.csv",
}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose how to load data:",
        ["📊 Use a pre-loaded stock", "📁 Upload my own CSV"],
        index=0
    )

    selected_stock = None
    uploaded_file  = None

    if data_source == "📊 Use a pre-loaded stock":
        selected_stock = st.selectbox(
            "Select a stock:",
            list(PRELOADED_STOCKS.keys()),
            index=0
        )
        st.caption("✅ Data fetched directly from the course GitHub repo — no upload needed.")
    else:
        uploaded_file = st.file_uploader("Historical Stock CSV (Kaggle format)", type=["csv"])
        st.caption("Expected columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`")

    st.markdown("---")
    st.markdown("**Project Details**")
    st.markdown("- Subject Code: `DA-304T`")
    st.markdown("- Semester: `6`")
    st.markdown("- Class: `MLDA / AIML-4 / AIML-5`")
    st.markdown("- Submission: `31 Mar 2026`")
    st.markdown("- Group Size: `3`")
    st.markdown("---")
    st.markdown("**Dataset Source**")
    st.markdown("[Kaggle – Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stockmarket-dataset)")

# ─────────────────────────────────────────────
# Load & Validate Data
# ─────────────────────────────────────────────
df         = None
stock_name = None

if data_source == "📊 Use a pre-loaded stock":
    url = PRELOADED_STOCKS[selected_stock]
    try:
        with st.spinner(f"Fetching {selected_stock} data from GitHub…"):
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        df         = pd.read_csv(io.StringIO(resp.text), parse_dates=['Date'])
        stock_name = selected_stock.split("–")[0].strip()
    except Exception as e:
        st.error(f"❌ Could not fetch data: {e}")
        st.stop()

else:
    if not uploaded_file:
        st.info("👈 Select a pre-loaded stock **or** upload your own CSV from the sidebar.")
        st.markdown("**Expected columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`")
        st.stop()
    df         = pd.read_csv(uploaded_file, parse_dates=['Date'])
    stock_name = uploaded_file.name.split('.')[0].upper()

df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

required_cols = {'Date','Open','High','Low','Close','Adj Close','Volume'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
st.markdown(f"### 📈 Ticker: `{stock_name}`  |  Rows loaded: `{len(df):,}`  |  Date range: `{df['Date'].min().date()}` → `{df['Date'].max().date()}`")

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — EDA & HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════
st.markdown("""<div class="phase-card">
<div class="phase-title">Phase 1 — Exploratory Data Analysis & Hypothesis Testing &nbsp; <span style="font-size:0.75rem;color:#64748b">(Unit I)</span></div>
</div>""", unsafe_allow_html=True)

# ── 1.1  Feature Engineering ──────────────────────────────────
st.markdown("#### 1.1  Feature Engineering")
df['Daily_Return']          = df['Adj Close'].pct_change() * 100
df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)
df['Up_Day']                = df['Daily_Return'] > 0

with st.expander("View sample dataset (first 10 rows)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ── 1.2  Descriptive Statistics ───────────────────────────────
st.markdown("#### 1.2  Descriptive Statistics")

dr  = df['Daily_Return'].dropna()
vol = df['Volume'].dropna()

stats_data = {
    "Metric":            ["Mean", "Variance", "Std Dev", "Skewness", "Kurtosis", "Min", "Max"],
    "Daily Return (%)":  [f"{dr.mean():.4f}", f"{dr.var():.4f}", f"{dr.std():.4f}",
                          f"{dr.skew():.4f}", f"{dr.kurt():.4f}", f"{dr.min():.4f}", f"{dr.max():.4f}"],
    "Volume":            [f"{vol.mean():,.0f}", f"{vol.var():,.0f}", f"{vol.std():,.0f}",
                          f"{vol.skew():.4f}", f"{vol.kurt():.4f}", f"{vol.min():,.0f}", f"{vol.max():,.0f}"],
}
st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# ── 1.3  Visualizations ───────────────────────────────────────
st.markdown("#### 1.3  Visualizations")

col_a, col_b = st.columns(2)

with col_a:
    # Adj Close time series
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df['Date'], df['Adj Close'], color=SKY, linewidth=1)
    ax.fill_between(df['Date'], df['Adj Close'], alpha=0.15, color=SKY)
    ax.set_title(f"{stock_name} — Adjusted Close Price", fontweight='bold')
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

with col_b:
    # Daily Return histogram
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.hist(dr, bins=60, color=AMBER, edgecolor='#0f172a', alpha=0.85)
    ax2.axvline(dr.mean(), color=ROSE, linestyle='--', linewidth=1.5, label=f"Mean = {dr.mean():.2f}%")
    ax2.axvline(dr.mean() + dr.std(), color=EMERALD, linestyle=':', linewidth=1.2, label=f"+1σ")
    ax2.axvline(dr.mean() - dr.std(), color=EMERALD, linestyle=':', linewidth=1.2, label=f"−1σ")
    ax2.set_title("Daily Return Distribution", fontweight='bold')
    ax2.set_xlabel("Daily Return (%)"); ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)

# Rolling volatility
fig3, ax3 = plt.subplots(figsize=(14, 3))
rolling_vol = dr.rolling(30).std()
ax3.plot(df['Date'].iloc[len(df)-len(rolling_vol):], rolling_vol, color=ROSE, linewidth=1)
ax3.set_title("30-Day Rolling Volatility (Std Dev of Daily Returns)", fontweight='bold')
ax3.set_xlabel("Date"); ax3.set_ylabel("Volatility (%)")
ax3.grid(True)
fig3.tight_layout()
st.pyplot(fig3)

# ── 1.4  Hypothesis Testing ───────────────────────────────────
st.markdown("#### 1.4  Hypothesis Testing — Volume on Up vs Down Days")

st.markdown("""
<div class="hyp-box">
  <strong>H₀ (Null Hypothesis):</strong> The mean trading volume on Up-days equals the mean trading volume on Down-days.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;μ_up = μ_down<br><br>
  <strong>H₁ (Alternative Hypothesis):</strong> The mean trading volume on Up-days is different from that on Down-days.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;μ_up ≠ μ_down (two-tailed)<br><br>
  <strong>Test Used:</strong> Welch's Two-Sample t-Test (does not assume equal variances) &nbsp;|&nbsp; <strong>Significance Level α:</strong> 0.05
</div>
""", unsafe_allow_html=True)

vol_up   = df[df['Up_Day']  == True ]['Volume'].dropna()
vol_down = df[df['Up_Day']  == False]['Volume'].dropna()
t_stat, p_value = stats.ttest_ind(vol_up, vol_down, equal_var=False)
cohen_d = (vol_up.mean() - vol_down.mean()) / np.sqrt((vol_up.std()**2 + vol_down.std()**2) / 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Up-Day Avg Volume",   f"{vol_up.mean()/1e6:.2f}M")
col2.metric("Down-Day Avg Volume", f"{vol_down.mean()/1e6:.2f}M")
col3.metric("T-Statistic",         f"{t_stat:.4f}")
col4.metric("P-Value",             f"{p_value:.6f}")

if p_value < 0.05:
    st.success(f"✅ **Reject H₀** — p = {p_value:.6f} < 0.05. There IS a statistically significant difference in trading volume between Up and Down days. Cohen's d = {cohen_d:.4f}")
else:
    st.warning(f"❌ **Fail to Reject H₀** — p = {p_value:.6f} ≥ 0.05. Insufficient evidence to conclude a difference in volume.")

col_c, col_d = st.columns(2)
with col_c:
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    bp = ax4.boxplot(
        [vol_up/1e6, vol_down/1e6],
        labels=["Up Days", "Down Days"],
        patch_artist=True,
        medianprops=dict(color='#0f172a', linewidth=2)
    )
    bp['boxes'][0].set_facecolor(EMERALD); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(ROSE);    bp['boxes'][1].set_alpha(0.7)
    ax4.set_title("Volume Distribution — Up vs Down Days", fontweight='bold')
    ax4.set_ylabel("Volume (Millions)")
    fig4.tight_layout()
    st.pyplot(fig4)

with col_d:
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.hist(vol_up/1e6,   bins=40, color=EMERALD, alpha=0.6, label="Up Days",   edgecolor='#0f172a')
    ax5.hist(vol_down/1e6, bins=40, color=ROSE,    alpha=0.6, label="Down Days", edgecolor='#0f172a')
    ax5.set_title("Volume Histogram — Up vs Down Days", fontweight='bold')
    ax5.set_xlabel("Volume (Millions)"); ax5.set_ylabel("Frequency")
    ax5.legend()
    fig5.tight_layout()
    st.pyplot(fig5)

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE OPTIMIZATION VIA EIGENVECTORS / PCA
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""<div class="phase-card">
<div class="phase-title">Phase 2 — Feature Optimization via Eigenvectors (PCA) &nbsp; <span style="font-size:0.75rem;color:#64748b">(Unit IV)</span></div>
</div>""", unsafe_allow_html=True)

features = ['Open', 'High', 'Low', 'Volume']
df_pca_raw = df[features].dropna()

# ── 2.1  Covariance Matrix ────────────────────────────────────
st.markdown("#### 2.1  Covariance Matrix")
st.markdown("Constructed on `Open`, `High`, `Low`, `Volume` columns (before scaling, for interpretability):")

cov_matrix = np.cov(df_pca_raw.T)
cov_df = pd.DataFrame(cov_matrix, columns=features, index=features)
st.dataframe(cov_df.style.background_gradient(cmap='Blues'), use_container_width=True)

# ── 2.2  Eigenvalues & Eigenvectors ──────────────────────────
st.markdown("#### 2.2  Eigenvalues & Eigenvectors")
st.markdown("""
We compute eigenvalues (λ) and eigenvectors of the **standardised** covariance matrix (correlation matrix)  
so that the high-variance Volume column does not dominate purely due to scale.
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca_raw)
cov_scaled = np.cov(X_scaled.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_scaled)
sort_idx    = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

eigen_df = pd.DataFrame(
    eigenvectors,
    index=features,
    columns=[f'PC{i+1}' for i in range(len(features))]
)
st.markdown("**Eigenvectors (Principal Component loadings):**")
st.dataframe(eigen_df.round(4), use_container_width=True)

explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained)

ev_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue':          eigenvalues.round(4),
    'Variance Explained (%)': (explained * 100).round(2),
    'Cumulative Variance (%)': (cumulative * 100).round(2),
})
st.dataframe(ev_df, use_container_width=True)

col_e, col_f = st.columns(2)
with col_e:
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    pcs = [f'PC{i+1}' for i in range(len(eigenvalues))]
    bars = ax6.bar(pcs, explained * 100, color=AMBER, edgecolor='#0f172a')
    ax6.set_title("Variance Explained by Each PC", fontweight='bold')
    ax6.set_ylabel("Explained Variance (%)")
    for b, v in zip(bars, explained * 100):
        ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}%',
                 ha='center', fontsize=9, color='white')
    fig6.tight_layout()
    st.pyplot(fig6)

with col_f:
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    ax7.plot(pcs, cumulative * 100, color=SKY, marker='o', linewidth=2)
    ax7.axhline(95, color=AMBER, linestyle='--', linewidth=1.2, label='95% threshold')
    ax7.set_title("Cumulative Variance Explained", fontweight='bold')
    ax7.set_ylabel("Cumulative Variance (%)")
    ax7.legend(); ax7.grid(True)
    fig7.tight_layout()
    st.pyplot(fig7)

# ── 2.3  Basis, Dimension & Interpretation ───────────────────
st.markdown("#### 2.3  Basis, Dimension & Portfolio Interpretation")

n_components_95 = int(np.searchsorted(cumulative, 0.95)) + 1

st.info(f"""
**Principal Component Analysis Results:**

- **PC1** (Eigenvalue = {eigenvalues[0]:.4f}) captures **{explained[0]*100:.1f}%** of total variance.  
  It is the *most informative* direction in feature space — a weighted combination of all four features  
  that retains maximum market information.
  
- **{n_components_95} principal component(s)** are sufficient to explain **≥ 95%** of the total variance,  
  reducing the original 4-dimensional feature space while preserving almost all information.

- **Basis & Dimension:** The eigenvectors form an **orthogonal basis** for the 4D feature space.  
  Each eigenvector is linearly independent (by construction), representing a distinct "market factor."  
  By projecting onto the top eigenvectors we obtain a **lower-dimensional subspace** (dimension = {n_components_95})  
  that captures the essential variance — ideal for feeding into regression models without multicollinearity.

- **Portfolio Diversification Insight:** Linear independence of eigenvectors ≡ uncorrelated market factors.  
  By building a portfolio whose exposures correspond to separate principal components,  
  an investor holds positions in mathematically orthogonal risk directions — true diversification.
""")

# PC1 vs PC2 scatter
X_pc = X_scaled @ eigenvectors
fig8, ax8 = plt.subplots(figsize=(7, 4))
sc = ax8.scatter(X_pc[:,0], X_pc[:,1], c=df_pca_raw.index, cmap='plasma', s=5, alpha=0.6)
ax8.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
ax8.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
ax8.set_title("PC1 vs PC2 — Projection of Trading Days", fontweight='bold')
plt.colorbar(sc, ax=ax8, label="Time (row index)")
fig8.tight_layout()
st.pyplot(fig8)

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — STATISTICAL MODELLING & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""<div class="phase-card">
<div class="phase-title">Phase 3 — Statistical Modelling & Regression Diagnostics &nbsp; <span style="font-size:0.75rem;color:#64748b">(Unit II)</span></div>
</div>""", unsafe_allow_html=True)

df_model = df.dropna(subset=['Open','High','Low','Volume','Adj Close','Target_Next_Day_Close']).copy()
X_cols = ['Open','High','Low','Volume']

# ── 3.1  Multiple Linear Regression ──────────────────────────
st.markdown("#### 3.1  Multiple Linear Regression — Predicting Next-Day Close")
st.markdown("""
Using the **Ordinary Least Squares** (geometry of least squares) formulation:  
`β = (XᵀX)⁻¹ Xᵀy`  
Features: `Open`, `High`, `Low`, `Volume` → Target: `Target_Next_Day_Close`
""")

X_reg = df_model[X_cols].values
y_reg = df_model['Target_Next_Day_Close'].values

# Manual OLS — geometry of least squares
X_b   = np.column_stack([np.ones(len(X_reg)), X_reg])   # add intercept column
beta_ols = np.linalg.lstsq(X_b, y_reg, rcond=None)[0]   # (XᵀX)⁻¹Xᵀy via LAPACK
y_pred_ols = X_b @ beta_ols

# Also fit sklearn for comparison
lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
y_pred = lin_reg.predict(X_reg)

# Metrics
mae  = mean_absolute_error(y_reg, y_pred)
rmse = np.sqrt(mean_squared_error(y_reg, y_pred))
r2   = r2_score(y_reg, y_pred)
n, p = len(y_reg), X_reg.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

coef_df = pd.DataFrame({
    "Feature":     ["Intercept"] + X_cols,
    "OLS β (manual)": beta_ols.round(6),
    "sklearn coef":   [lin_reg.intercept_] + list(lin_reg.coef_),
})
st.dataframe(coef_df, use_container_width=True)
st.caption("Manual OLS and sklearn coefficients should match — confirming correct implementation.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",        f"{r2:.4f}")
c2.metric("Adj. R²",   f"{adj_r2:.4f}")
c3.metric("MAE",       f"${mae:.4f}")
c4.metric("RMSE",      f"${rmse:.4f}")

# Predicted vs Actual
fig9, ax9 = plt.subplots(figsize=(14, 4))
ax9.plot(df_model['Date'].values, y_reg,  color=SKY,  linewidth=1,   label='Actual')
ax9.plot(df_model['Date'].values, y_pred, color=AMBER, linewidth=1, alpha=0.8, label='Predicted')
ax9.set_title("Multiple Linear Regression — Actual vs Predicted Next-Day Close", fontweight='bold')
ax9.set_xlabel("Date"); ax9.set_ylabel("Price (USD)")
ax9.legend(); ax9.grid(True)
fig9.tight_layout()
st.pyplot(fig9)

# ── 3.2  Regression Diagnostics ──────────────────────────────
st.markdown("#### 3.2  Regression Diagnostics")

residuals = y_reg - y_pred
std_resid = (residuals - residuals.mean()) / residuals.std()

# Cook's Distance (influence diagnostic)
# Cook's D ≈ (ê²ᵢ / (p * MSE)) × (hᵢᵢ / (1 - hᵢᵢ)²)
H = X_b @ np.linalg.pinv(X_b.T @ X_b) @ X_b.T   # hat matrix
h = np.diag(H)                                      # leverage
mse = np.mean(residuals**2)
cooks_d = (residuals**2 * h) / (p * mse * (1 - h)**2)
cooks_threshold = 4 / n

outlier_mask = cooks_d > cooks_threshold
n_outliers   = outlier_mask.sum()

st.markdown(f"**Cook's Distance Threshold:** 4/n = {cooks_threshold:.4f} &nbsp;|&nbsp; **Influential observations identified:** `{n_outliers}` ({n_outliers/n*100:.1f}%)")

fig10 = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig10, hspace=0.45, wspace=0.35)

# Residuals vs Fitted
ax_rv = fig10.add_subplot(gs[0, 0])
ax_rv.scatter(y_pred[~outlier_mask], residuals[~outlier_mask], color=SKY,  alpha=0.4, s=8)
ax_rv.scatter(y_pred[outlier_mask],  residuals[outlier_mask],  color=ROSE, alpha=0.8, s=20, label='Influential')
ax_rv.axhline(0, color=AMBER, linestyle='--', linewidth=1.5)
ax_rv.set_title("Residuals vs Fitted", fontweight='bold'); ax_rv.set_xlabel("Fitted"); ax_rv.set_ylabel("Residual")
ax_rv.legend(fontsize=8)

# QQ Plot
ax_qq = fig10.add_subplot(gs[0, 1])
(osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist='norm')
ax_qq.scatter(osm, osr, color=AMBER, alpha=0.5, s=8)
ax_qq.plot(osm, slope * np.array(osm) + intercept, color=ROSE, linewidth=1.5)
ax_qq.set_title("Normal Q-Q Plot", fontweight='bold'); ax_qq.set_xlabel("Theoretical Quantiles"); ax_qq.set_ylabel("Sample Quantiles")

# Scale-Location
ax_sl = fig10.add_subplot(gs[0, 2])
ax_sl.scatter(y_pred, np.sqrt(np.abs(std_resid)), color=EMERALD, alpha=0.4, s=8)
ax_sl.set_title("Scale-Location", fontweight='bold'); ax_sl.set_xlabel("Fitted"); ax_sl.set_ylabel("√|Std Residual|")

# Cook's Distance
ax_cd = fig10.add_subplot(gs[1, :2])
ax_cd.bar(range(len(cooks_d)), cooks_d, color=np.where(outlier_mask, ROSE, SKY), width=1, alpha=0.7)
ax_cd.axhline(cooks_threshold, color=AMBER, linestyle='--', linewidth=1.5, label=f"Threshold = {cooks_threshold:.4f}")
ax_cd.set_title(f"Cook's Distance — Influence Diagnostics ({n_outliers} influential points)", fontweight='bold')
ax_cd.set_xlabel("Observation Index"); ax_cd.set_ylabel("Cook's D")
ax_cd.legend(fontsize=8)

# Residual histogram
ax_rh = fig10.add_subplot(gs[1, 2])
ax_rh.hist(std_resid, bins=50, color=AMBER, edgecolor='#0f172a', alpha=0.8)
ax_rh.set_title("Standardised Residuals", fontweight='bold'); ax_rh.set_xlabel("Std Residual"); ax_rh.set_ylabel("Count")

fig10.patch.set_facecolor('#0f172a')
st.pyplot(fig10)

# Top influential days
if n_outliers > 0:
    st.markdown("**Top 10 Most Influential Days (Potential Market Crash / Extreme Events):**")
    influence_df = df_model[outlier_mask][['Date','Open','High','Low','Adj Close','Volume']].copy()
    influence_df['Cook_D']    = cooks_d[outlier_mask]
    influence_df['Residual']  = residuals[outlier_mask]
    st.dataframe(
        influence_df.sort_values('Cook_D', ascending=False).head(10).reset_index(drop=True),
        use_container_width=True
    )

# ── 3.3  Logistic Regression ──────────────────────────────────
st.markdown("#### 3.3  Logistic Regression — Predicting Next-Day Direction (Up / Down)")

df_model['Target_Up'] = (df_model['Target_Next_Day_Close'] > df_model['Adj Close']).astype(int)
X_log = df_model[X_cols].values
y_log = df_model['Target_Up'].values

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_log, y_log)
y_prob_log = log_reg.predict_proba(X_log)[:, 1]
y_pred_log = log_reg.predict(X_log)

logreg_coef_df = pd.DataFrame({
    "Feature":     X_cols,
    "Coefficient": log_reg.coef_[0].round(6),
    "Odds Ratio":  np.exp(log_reg.coef_[0]).round(4),
})
st.dataframe(logreg_coef_df, use_container_width=True)
st.markdown(f"**Intercept:** `{log_reg.intercept_[0]:.6f}`")

# Classification report
report = classification_report(y_log, y_pred_log, output_dict=True)
rep_df = pd.DataFrame(report).T.round(3)
st.markdown("**Classification Report:**")
st.dataframe(rep_df, use_container_width=True)

col_g, col_h = st.columns(2)

with col_g:
    # Confusion Matrix
    cm = confusion_matrix(y_log, y_pred_log)
    fig11, ax11 = plt.subplots(figsize=(5, 4))
    im = ax11.imshow(cm, cmap='Blues')
    ax11.set_xticks([0,1]); ax11.set_yticks([0,1])
    ax11.set_xticklabels(['Pred Down','Pred Up']); ax11.set_yticklabels(['Actual Down','Actual Up'])
    for i in range(2):
        for j in range(2):
            ax11.text(j, i, str(cm[i,j]), ha='center', va='center',
                      color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
    ax11.set_title("Confusion Matrix", fontweight='bold')
    fig11.tight_layout()
    st.pyplot(fig11)

with col_h:
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_log, y_prob_log)
    roc_auc = auc(fpr, tpr)
    fig12, ax12 = plt.subplots(figsize=(5, 4))
    ax12.plot(fpr, tpr, color=AMBER, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax12.plot([0,1],[0,1], color=SLATE, linestyle='--', linewidth=1)
    ax12.set_title("ROC Curve", fontweight='bold')
    ax12.set_xlabel("False Positive Rate"); ax12.set_ylabel("True Positive Rate")
    ax12.legend(); ax12.grid(True)
    fig12.tight_layout()
    st.pyplot(fig12)

# Predicted probability over time
fig13, ax13 = plt.subplots(figsize=(14, 3))
ax13.plot(df_model['Date'].values, y_prob_log, color=AMBER, linewidth=0.8, alpha=0.8)
ax13.axhline(0.5, color=ROSE, linestyle='--', linewidth=1, label='Decision boundary (0.5)')
ax13.fill_between(df_model['Date'].values, y_prob_log, 0.5,
                  where=(y_prob_log >= 0.5), color=EMERALD, alpha=0.2, label='Predicted Up')
ax13.fill_between(df_model['Date'].values, y_prob_log, 0.5,
                  where=(y_prob_log < 0.5),  color=ROSE,    alpha=0.2, label='Predicted Down')
ax13.set_title("Logistic Regression — Predicted Probability of Next-Day Price Increase", fontweight='bold')
ax13.set_xlabel("Date"); ax13.set_ylabel("P(Up)")
ax13.legend(fontsize=8); ax13.grid(True)
fig13.tight_layout()
st.pyplot(fig13)

# ── Preview table ──────────────────────────────────────────────
with st.expander("📋 Preview prediction table (first 10 rows)"):
    preview = df_model[['Date','Adj Close','Target_Next_Day_Close']].copy()
    preview['LinReg_Pred']   = y_pred
    preview['LogReg_P(Up)']  = y_prob_log.round(4)
    preview['LogReg_Signal'] = np.where(y_prob_log >= 0.5, '⬆ Up', '⬇ Down')
    st.dataframe(preview.head(10).reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.success("🎓 **Analysis complete.** All three project phases executed — EDA (Unit I), PCA/Eigenvectors (Unit IV), and Statistical Modelling with Diagnostics (Unit II).")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; margin-top:1rem;">
  MAHARAJA AGRASEN INSTITUTE OF TECHNOLOGY &nbsp;|&nbsp; DA-304T &nbsp;|&nbsp; AY 2025-26
</div>
""", unsafe_allow_html=True)
