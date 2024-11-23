
---

# Time Series Forecasting & Volatility Modeling

This repository showcases various models and techniques for time series forecasting and volatility modeling using Python. It includes the use of **ARIMA**, **SARIMA**, and **ARCH/GARCH** models to analyze electricity consumption data and financial returns. The models aim to identify trends, seasonality, and volatility clustering, along with providing insights for forecasting future values.

## Table of Contents

- [Overview](#overview)
- [Data Analysis](#data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [ARIMA](#arima)
  - [SARIMA](#sarima)
  - [ARCH](#arch)
  - [GARCH](#garch)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Overview

This repository analyzes time series data, primarily focusing on electricity consumption and financial returns, to model trends and volatility. The models used in this project are:

- **ARIMA** (AutoRegressive Integrated Moving Average) for modeling non-seasonal time series data.
- **SARIMA** (Seasonal ARIMA) for incorporating seasonality in the time series data.
- **ARCH** (Autoregressive Conditional Heteroskedasticity) for modeling time-varying volatility in financial returns.
- **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) for more advanced volatility modeling, considering both short- and long-term volatility.

## Data Analysis

The primary dataset used in this project is daily electricity consumption data consisting of 630 observations. This data exhibits seasonality and trends, which are modeled using ARIMA and SARIMA models.

## Data Preprocessing

- **Read CSV file** containing energy consumption data and parse datetime columns (`date` and `time`).
- **Sort** the DataFrame by datetime for sequential processing.
- **Calculate daily consumption** using a rolling sum over a 1-day window.
- **Calculate monthly statistics** such as total, mean, and count of consumption.
- **Filter day and night consumption** based on time and consumption thresholds (greater than 0.5).

### Exploratory Data Analysis (EDA)

- **Seasonality Analysis**: Graphs were plotted to observe seasonal consumption patterns. We explored the differences in consumption across months and seasons.
- **Residual Analysis**: The residuals from the models were analyzed to ensure no patterns exist and that they behave as white noise.

## Modeling

### ARIMA (AutoRegressive Integrated Moving Average)

ARIMA models are used for forecasting time series data that does not show seasonality. In this project, an **ARIMA(2, 0, 0)** model was fitted to the electricity consumption data.

- **Model Summary**:
    - **AIC**: 2809.229
    - **BIC**: 2827.012
    - **RMSE**: 2.092
    - **MSE**: 4.374

```python
print(model_fit.summary())
```

- **Key Insights**:
    - The AR coefficients (ar.L1 = 0.5632, ar.L2 = 0.2350) show significant influence on the current consumption.
    - Residuals show no significant autocorrelation (Ljung-Box test p-value = 0.96), suggesting that the model has adequately captured the underlying structure of the data.

### SARIMA (Seasonal ARIMA)

The SARIMA model was used to incorporate both seasonality and trend into the forecasting model. A **SARIMA(2, 0, 0)x(1, 1, [1], 12)** model was fitted to the data.

- **Model Summary**:
    - **AIC**: 2800.605
    - **BIC**: 2822.738
    - **RMSE**: 2.101
    - **MSE**: 4.413

```python
print(sarima_model_fit.summary())
```

- **Key Insights**:
    - Seasonal AR and MA components (ar.S.L12 and ma.S.L12) were used to capture periodic fluctuations, with significant coefficients.
    - High kurtosis in the residuals suggests the model might not fully capture the distribution of the errors, which may indicate the need for further refinement.

### ARCH (Autoregressive Conditional Heteroskedasticity)

The **ARCH** model was applied to capture volatility clustering in the financial returns data.

- **Model Summary**:
    - **AIC**: 6480.15
    - **BIC**: 6493.48
    - **Volatility Model**: omega = 1727.9079, alpha[1] = 0.0000

```python
print(arch_model_fit.summary())
```

- **Key Insights**:
    - The ARCH model revealed weak conditional heteroskedasticity as the alpha[1] coefficient was statistically insignificant (p-value = 1.000), indicating minimal volatility clustering.
    - Despite the weak significance, the model still validates the presence of ARCH behavior in the data.

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

The **GARCH** model was applied for more advanced volatility modeling, capturing both short-term and long-term effects.

- **Key Insights**:
    - GARCH models are preferred when there are both short-term and long-term volatility patterns.
    - This model enhances the ability to forecast volatility in financial markets, especially with more complex volatility dynamics.

## Results

### Visualization
- **Residual Plots**: Visual comparison of actual and fitted values to assess model accuracy.
- **KDE Visualization**: To inspect the distribution of residuals and identify any non-random patterns.

### Model Performance
- The **RMSE** and **MSE** metrics show the error levels for each model. The ARIMA and SARIMA models performed well, although further refinement is needed for better forecasting accuracy.
- The **ARCH** and **GARCH** models provide insights into the volatility dynamics, though the results indicate that more sophisticated modeling might be required for complex financial data.

## Installation

To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/rtpunati/Electricity-Forecast.git
   ```

2. Install the required dependencies:
   ```bash
pip install pandas matplotlib seaborn statsmodels pmdarima numpy sklearn arch
   ```

## Usage

1. Load the dataset and prepare the data for modeling.
2. Fit ARIMA, SARIMA, ARCH, and GARCH models using the provided Python scripts.
3. Visualize the results and residuals to assess model performance.
4. Use the trained models for forecasting future time points or for volatility analysis.

### Example:

```python
import statsmodels.api as sm

# ARIMA model fitting
model = sm.tsa.ARIMA(data, order=(2, 0, 0))
model_fit = model.fit()

# SARIMA model fitting
sarima_model = sm.tsa.SARIMAX(data, order=(2, 0, 0), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()

# ARCH model fitting
from arch import arch_model
arch_model = arch_model(returns, vol='ARCH')
arch_fit = arch_model.fit()
```

## References

1. **ARIMA Model**: Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2008). Time Series Analysis: Forecasting and Control.
2. **SARIMA Model**: Hyndman, R.J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
3. **ARCH/GARCH Models**: Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation.

---
