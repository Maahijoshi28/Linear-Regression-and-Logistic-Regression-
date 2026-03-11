# Real-Estate Intelligence System: A Hybrid Regression Project

### 🚀 Overview
This project implements a dual-model machine learning pipeline designed to analyze real estate data. It combines **Linear Regression** and **Logistic Regression** to transform raw property data into actionable investment insights.

### 🧠 How It Works
The system follows a "Predict-then-Decide" architecture:
1.  **Price Estimation (Linear Regression):** Predicts the fair market value (continuous numerical output) of a house based on features like square footage, age, and bedrooms.
2.  **Investment Classification (Logistic Regression):** Uses the delta between the actual price and predicted price to classify properties as a **"Bargain"** (1) or **"Overpriced"** (0).

---

### 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

---

### 📊 Methodology & Math

#### Linear Regression
The model minimizes the **Sum of Squared Errors (SSE)** to find the best-fit hyperplane for price prediction:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$$

#### Logistic Regression
The system applies the **Sigmoid Function** to map the input features into a probability between 0 and 1 for classification:

$$P(y=1) = \frac{1}{1 + e^{-z}}$$

---

### 📈 Evaluation Results
The project is evaluated using two sets of metrics:
* **Regression:** $R^2$ Score and Mean Absolute Error (MAE).
* **Classification:** Accuracy Score and a **Confusion Matrix** to visualize True Positives (correctly identified deals).



---

### 💻 How to Run
1. Clone this repository.
2. Open the `.ipynb` file in **Google Colab**.
3. Run all cells to see the synthetic data generation, model training, and final Results Table.

---
*Developed as part of my portfolio in Computer Science Engineering (AIML).*
