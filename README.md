# Predictive-Paradox
# Hourly Electricity Demand Forecasting

A professional machine learning pipeline for **one-hour-ahead electricity demand forecasting** using **XGBoost**, built on nearly a decade of historical grid telemetry and weather data.

This project focuses on forecasting `demand_mw` at time `t+1` using a robust classical tabular modeling approach. Beyond model training, the project emphasizes **real-world data cleaning, anomaly handling, temporal feature engineering, and leakage-free time-series evaluation**.

## Project Overview

Electricity demand forecasting is a critical problem for grid planning and operational stability. This project develops a reliable forecasting pipeline that predicts hourly grid demand one hour ahead using historical demand signals and aligned weather data.

A major strength of the work lies in handling messy, real-world datasets. The pipeline was designed not only to achieve strong predictive performance, but also to systematically resolve data quality issues that would otherwise make the model unreliable.

## Objectives

- Forecast hourly electricity demand (`demand_mw`) one hour ahead.
- Build a strong classical machine learning baseline using XGBoost.
- Clean and reconstruct anomalous target values in long-range historical grid data.
- Merge electricity and weather datasets without temporal misalignment.
- Engineer time-aware features while preventing data leakage.
- Evaluate model performance on a strictly unseen chronological test set.

## Dataset and Problem Setting

The modeling task uses nearly ten years of historical electricity demand, grid telemetry, and weather observations. The target variable is hourly electricity demand measured in megawatts (`demand_mw`).

The prediction setup is a **one-step-ahead forecasting problem**, where the model uses information available up to time `t` to predict demand at time `t+1`.

## Data Cleaning and Preprocessing

### Target Anomaly Handling

As expected in real-world grid data, the target series contained severe anomalies, including:

- Unrealistic drops to values as low as **6 MW**.
- Extreme spikes reaching **121,000 MW**.

A static percentile filter was not suitable because electricity demand evolves over time, and a fixed global cutoff would unfairly distort more recent years with naturally higher demand levels.

To address this, a **dynamic rolling Z-score approach** was used:

1. Compute a **7-day rolling mean** and **7-day rolling standard deviation**.
2. Calculate the rolling Z-score for each observation.
3. Flag any point with **Z-score > 3.5** as anomalous.
4. Replace anomalous values with missing values.
5. Reconstruct them using short-range **time-based linear interpolation**.

This method preserved long-term demand growth trends while still removing unrealistic local spikes and drops.

### Domain-Driven Cleaning Rules

Additional data corrections were guided by domain knowledge:

- **Solar generation** was forced to zero during nighttime hours (**7 PM to 6 AM**).
- Missing **wind generation** values were treated as zero.
- Missing **import** values were also treated as zero, assuming system inactivity rather than sensor failure.

These rules improved consistency and reduced ambiguity in telemetry interpretation.

## Weather Data Integration

Merging electricity demand data with weather data was one of the most important preprocessing challenges.

An initial direct merge produced widespread missing weather values. The issue was traced back to inconsistent datetime formats:

- The weather dataset used a **European-style string datetime format**.
- The demand dataset was already stored in a standard datetime representation.

To fix the mismatch:

- Both datasets were explicitly converted using `pd.to_datetime()`.
- Flexible parsing was applied to handle inconsistent string formats.
- Timestamps were rounded to the nearest hour.
- Temporal alignment was verified before merging.

Once the datetime formats were standardized, the merge succeeded without the earlier large-scale data loss.

## Feature Engineering

Because classical tabular models do not automatically understand time structure, temporal patterns had to be encoded explicitly.

### Calendar Features

The following time-based calendar features were created:

- Hour of day
- Day of week
- Month
- Weekend indicator

These features help the model learn regular consumption cycles such as daily peaks, weekday effects, and seasonal variation.

### Lag Features

To provide historical context, lagged demand features were added, including:

- `lag_1` — demand 1 hour earlier
- `lag_2` — demand 2 hours earlier
- `lag_24` — demand 24 hours earlier
- `lag_168` — demand 168 hours earlier (weekly seasonality)

These lags allow the model to capture short-term persistence as well as recurring daily and weekly patterns.

### Rolling Statistics

Rolling statistical features were also engineered:

- 24-hour moving average
- 24-hour moving standard deviation

To prevent leakage, all rolling features were **shifted by one hour**, ensuring that no future information was used when generating predictors.

## Missing Value Strategy

Feature engineering naturally introduced missing values, especially at the beginning of the series where lagged observations were unavailable.

A naive strategy of dropping all rows with any missing value would have removed almost the entire dataset. Instead, a selective approach was used:

- Forward-fill missing weather observations to preserve continuity.
- Remove only the earliest rows affected by the maximum lag window.
- Specifically discard the initial **168 hours** required by the longest lag feature.

This preserved the vast majority of the data while keeping the training set valid and leakage-free.

## Train-Test Split

Since this is a time-series forecasting task, a random train-test split would have caused data leakage by exposing the model to future patterns during training.

A strict chronological split was used instead:

- **Training set:** all observations up to the end of **2023**
- **Test set:** all observations from **2024**

The 2024 holdout period contained **8,784 hourly observations**, reflecting the fact that 2024 was a leap year. This created a realistic and fully unseen test scenario.

## Model

The final forecasting model was an **XGBoost regressor**, selected for its strong performance on structured tabular data.

To keep the feature space focused and reduce unnecessary complexity, redundant generation-related variables were removed. The final model relied primarily on:

- Historical demand features
- Time-based calendar features
- Weather variables

This helped the model balance predictive power with interpretability and robustness.

## Performance

The model performed strongly on the unseen 2024 test set.

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | **186.9 MW** |
| Root Mean Squared Error (RMSE) | **262.3 MW** |
| Mean Absolute Percentage Error (MAPE) | **1.69%** |

These results indicate that the pipeline achieved high forecasting accuracy for a real-world electricity demand problem.

## Visual Validation

To assess predictions qualitatively, actual versus predicted demand was plotted for the **first two weeks of 2024**.

The forecast closely tracked:

- Regular daily load cycles
- Expected peak-demand behavior
- Unexpected short-term fluctuations

This visual alignment reinforced the quantitative evaluation and demonstrated that the model generalized well to unseen operational data.

## Key Takeaways

- Real-world forecasting performance depends heavily on data quality, not just model choice.
- Dynamic anomaly detection is more reliable than static global filtering when long-term trends are present.
- Datetime standardization is essential before merging independent time-series datasets.
- Lag and rolling features can make classical models highly competitive for demand forecasting.
- Strict chronological validation is necessary to obtain trustworthy time-series results.

