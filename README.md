[Project Link](https://ssmdaprojectarpitrawat.streamlit.app/)
## **Objective**

* Analyze historical daily stock prices for a major publicly traded company.
* Apply **exploratory data analysis (EDA)** to understand stock behavior and volatility.
* Use **linear algebra (Eigenvectors & Principal Component Analysis)** to reduce dimensionality in correlated features.
* Build **statistical models** (linear and logistic regression) to predict the next day’s stock price and direction.

---

## **Dataset**

* Source: [Kaggle – Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stockmarket-dataset)
* CSV Columns:

  * `Date` – trading date
  * `Open` – opening price
  * `High` – maximum price
  * `Low` – minimum price
  * `Close` – closing price
  * `Adj Close` – adjusted closing price (splits & dividends)
  * `Volume` – number of shares traded

**Additional calculated features:**

* `Daily_Return` – percentage change in adjusted close
* `Target_Next_Day_Close` – next day’s closing price

---

## **Project Phases**

### **Phase 1: Exploratory Data Analysis (EDA) & Hypothesis Testing**

* **Data Preparation:** Calculate `Daily_Return` and `Target_Next_Day_Close`.
* **Descriptive Statistics:** Mean, variance, standard deviation of `Volume` and `Daily_Return`.
* **Visualization:** Time-series plot of `Adj Close`.
* **Hypothesis Testing:** Check if trading volume differs significantly on days the stock goes up vs. down.

---

### **Phase 2: Feature Optimization via Eigenvectors**

* **Correlation Handling:** Open, High, Low, and Close are highly correlated.
* **Covariance Matrix:** Construct from Open, High, Low, and Volume.
* **Eigenvalues & Eigenvectors:** Identify the **principal components**.
* **Dimensionality Reduction:** Reduce correlated features into independent basis vectors to improve regression efficiency.

---

### **Phase 3: Statistical Modelling & Diagnostics**

* **Multiple Linear Regression:** Predict `Target_Next_Day_Close` from Open, High, Low, and Volume.

  * Compute regression coefficients using **least squares geometry**.
* **Logistic Regression:** Predict if the next day’s price goes **up (1)** or **down (0)**.
* **Diagnostics:**

  * Plot residuals
  * Calculate influence diagnostics to identify outliers like market crash days

---

## **Tools & Libraries**

* Python, Pandas, NumPy
* Matplotlib / Seaborn for visualization
* Scikit-learn for regression and logistic regression

---

## **Highlights / Learning Outcomes**

* Understand **historical stock price patterns and volatility**
* Apply **linear algebra and PCA for feature reduction**
* Build **predictive statistical models** for regression and classification
* Perform **regression diagnostics** to handle real-world data issues like outliers
