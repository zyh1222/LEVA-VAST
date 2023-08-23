import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import linregress

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import stats

def check_compatibility_with_power_law(x, y, significance_level=0.05):
    """Check if y follows power law distribution with respect to x."""
    initial_beta_guess = 0.7
    initial_a_guess = 1
    min_beta = 0.4
    params, _ = curve_fit(lambda x, a, beta: power_law(x, a, beta), x, y, p0=[initial_a_guess, initial_beta_guess],maxfev=1000)
    a = params[0]
    b = params[1]
    print(y,a,b)
    # Check if beta is within a certain range
    if b < min_beta:
        return True
    predicted = power_law(x, a, -b)

    # Calculate residuals
    residuals = y - predicted
    mu, std = stats.norm.fit(residuals)

    # Calculate the p-values for all residuals
    p_values = 1 - stats.norm(mu, std).cdf(np.abs(residuals))
    # If any value has a p-value less than the significance level, it doesn't follow power law
    return not any(p < significance_level for p in p_values)

def get_outstanding_topn(data):
    sorted_indices = np.argsort(data)[::-1]
    outstanding_indices = []
    n = 1
    length = len(data)
    
    while n < length:
        x = np.array(range(1, length - n + 1))
        y = [data[i] for i in sorted_indices[n:]]

        # Ensure we have enough points to fit the model
        if len(y) < 2:  
            return [sorted_indices[i] for i in range(n)]

        # Check if y follows power law distribution
        if not check_compatibility_with_power_law(x, y):
            # Store the range of indices that don't follow power law
            outstanding_indices.append(sorted_indices[n-1])
        else:
            # If we encounter a range that follows power law, break the loop
            outstanding_indices.append(sorted_indices[n-1])
            break
        n += 1

    return outstanding_indices if outstanding_indices else None
def get_outstanding_no1(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):

    result, p_value = calculate_outstanding_no1_significance(measure_values)
    score_significance = 1 - p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)
    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Outstanding No1",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }

    return item


def get_outstanding_top2(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):

    result, p_value = calculate_outstanding_top2_significance(measure_values)
    score_significance = 1 - p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)
    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Outstanding Top2",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }

    return item


def calculate_outstanding_top2_significance(data):
    # Sort data in descending order
    sorted_data = np.sort(data)[::-1]

    # Conduct regression analysis using power-law function for all values excluding top2
    x = np.array(range(1, len(sorted_data)-1))  # Excluding top 2 values
    y = sorted_data[2:]  # Excluding top 2 values

    params, _ = curve_fit(power_law, x, y)
    alpha = params[0]

    # Calculate the residuals
    residuals = y - power_law(x, alpha)

    # Train a Gaussian model on the residuals
    mu, std = stats.norm.fit(residuals)

    # Predict top2 values and get the corresponding residuals R1 and R2
    top2_values = sorted_data[:2]
    # 1 and 2 because top2 are the first and second in the sorted list
    top2_predicted = power_law(np.array([1, 2]), alpha)
    R1, R2 = top2_values - top2_predicted

    # Calculate the p-values for R1 and R2 given the Gaussian model
    p_value1 = 1 - stats.norm(mu, std).cdf(R1)
    p_value2 = 1 - stats.norm(mu, std).cdf(R2)

    # Return the minimum p-value as we are testing a joint hypothesis (both values are outstanding)
    return top2_values, min(p_value1, p_value2)


def get_outstanding_last(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):
    result, p_value = calculate_outstanding_last_significance(measure_values)
    score_significance = 1 - p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)
    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Outstanding Last",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def get_evenness(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):

    result, p_value = calculate_evenness_significance(measure_values)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Evenness",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_evenness_significance(data):

    mean = np.mean(data)
    # Define an expected distribution where all values are the same (i.e., perfect evenness)
    expected_distribution = [mean] * len(data)

    # Conduct a chi-square test to get a p-value
    chisq, p_value = chisquare(data, expected_distribution)

    return mean, p_value
    # The p-value indicates whether the observed distribution significantly deviates from perfect evenness


def power_law(x, alpha, beta = 0.7):
    return alpha * (x ** - beta)


def calculate_outstanding_no1_significance(data):
    # Sort data in descending order
    sorted_data = np.sort(data)[::-1]

    # Conduct regression analysis using power-law function
    x = np.array(range(1, len(sorted_data)))  # Excluding max value
    y = sorted_data[1:]  # Excluding max value

    params, _ = curve_fit(power_law, x, y)
    alpha = params[0]

    # Generate the fitted curve using the fitted parameters
    y_fit = power_law(x, alpha)

    # Plot the original data and the fitted curve
    # plt.scatter(x, y, label='Original Data')
    # plt.plot(x, y_fit, label='Fitted Curve', color='red')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Calculate the residuals
    residuals = y - power_law(x, alpha)

    # Train a Gaussian model on the residuals
    mu, std = stats.norm.fit(residuals)

    # Predict xmax and get the corresponding residual R
    xmax = sorted_data[0]
    # 1 because xmax is the first in the sorted list
    xmax_predicted = power_law(1, alpha)
    R = xmax - xmax_predicted

    # Calculate the p-value for R given the Gaussian model
    p_value = 1 - stats.norm(mu, std).cdf(R)

    return xmax, p_value


def calculate_outstanding_last_significance(data):
    # Sort data in ascending order
    sorted_data = np.sort(data)

    # Conduct regression analysis using power-law function
    x = np.array(range(1, len(sorted_data)))  # Excluding min value
    y = sorted_data[1:]  # Excluding min value

    params, _ = curve_fit(power_law, x, y)
    alpha = params[0]

    # Calculate the residuals
    residuals = y - power_law(x, alpha)

    # Train a Gaussian model on the residuals
    mu, std = stats.norm.fit(residuals)

    # Predict xmin and get the corresponding residual R
    xmin = sorted_data[0]
    # 1 because xmin is the first in the sorted list
    xmin_predicted = power_law(1, alpha)
    R = xmin - xmin_predicted

    # Calculate the p-value for R given the Gaussian model
    p_value = 1 - stats.norm(mu, std).cdf(R)

    return xmin, p_value


def calculate_final_score(df, subspace_df, score_significance, omega_s, omega_c):
    # if subspace:
    #     # Check if subspace is a single tuple or multiple tuples
    #     if isinstance(outstanding_no_1, tuple):
    #         # Multi-tuple case
    #         context_filter = (df[list(subspace)] == outstanding_no_1[:len(subspace)]).all(axis=1)
    #     else:
    #         # Single-tuple case
    #         context_filter = df[subspace] == outstanding_no_1
    #     print(2222, context_filter)

    #     score_context = df.loc[context_filter].shape[0] / df.shape[0]
    # else:
    score_context = subspace_df.shape[0] / df.shape[0]
    score = omega_s * score_significance + omega_c * score_context
    return score, score_context


def get_change_point(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):
    result, p_value = calculate_change_point_significance(measure_values)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Change Point",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_change_point_significance(data):

    data = data.sort_index()
    data['Change'] = data.pct_change(
    ) * 100
    change_point = data['Change'].idxmax()
    change_point_index = data['Change'].argmax()
    y_left = data['Change'][:change_point_index - 1]
    y_right = data['Change'][change_point_index + 1:]

    # Calculate means
    y_left_mean = sum(y_left) / len(y_left)
    y_right_mean = sum(y_right) / len(y_right)

    # Calculate standard deviation of the mean
    y = y_left + y_right
    sigma_y = (sum(y_i ** 2 for y_i in y) / (2 * len(y)) -
               (sum(y) / (2 * len(y))) ** 2) ** 0.5
    sigma_mu_y = sigma_y / (len(y) ** 0.5)

    # Calculate k_mean
    k_mean = abs(y_left_mean - y_right_mean) / sigma_mu_y

    # Calculate p-value
    p_value = 1 - norm.cdf(k_mean)

    return change_point, p_value


def get_outlier(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):
    result, p_value = calculate_outlier_significance(measure_values)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "type": "Outlier",
        "focus": result,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_outlier_significance(data):
    # Sort data in ascending order
    sorted_data = data.sort_values()

    # Calculate mean and standard deviation
    mu, std = norm.fit(sorted_data)

    # Define a threshold for outlier detection (e.g., 3 standard deviations from the mean)
    outlier_threshold = mu + 3 * std

    # Detect potential outliers
    potential_outliers = sorted_data[sorted_data > outlier_threshold]

    # Calculate p-values for potential outliers
    outlier_p_values = 1 - norm.cdf((potential_outliers - mu) / std)

    # Return the potential outlier with the lowest p-value
    outlier = potential_outliers[outlier_p_values.argmin()]
    p_value = outlier_p_values.min()

    return outlier, p_value


def get_trend(df, subspace_df, subspace, breakdown_dim, measure_values, omega_s, omega_c):
    trend_direction, p_value = calculate_trend_significance(
        measure_values)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "parameter": trend_direction,
        "type": "Trend",
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_trend_significance(data):
    # Calculate the cumulative sum of the data
    data_cumsum = np.cumsum(data)

    # Create an array with the indices of the data
    x = np.array(range(len(data_cumsum)))

    # Apply linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, data_cumsum)

    # Trend direction
    trend_direction = "increase" if slope > 0 else "decrease"

    return trend_direction, p_value


def get_seasonality(df, subspace_df, subspace, breakdown_dim, measure_values, freq, omega_s, omega_c):
    trend_direction, interval, p_value = calculate_seasonality_significance(
        measure_values)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "parameter": trend_direction,
        "type": "Seasonality",
        "focus": interval,
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_seasonality_significance(data, freq):
    # Apply seasonal decomposition
    decomp = seasonal_decompose(data, model='additive', freq=freq)
    seasonal = decomp.seasonal

    # Estimate the seasonality interval
    interval = [seasonal[0], seasonal[-1]]

    # Check if the series increases or decreases
    trend = "increase" if (seasonal[-1] - seasonal[0]) > 0 else "decrease"

    # Apply Ljung-Box test to check for autocorrelation in residuals up to lag `freq`
    _, p_value = acorr_ljungbox(decomp.resid.dropna(), lags=[freq])

    return trend, interval, p_value[0]


def get_correlation(df, subspace_df, subspace, breakdown_dim, measure_1, measure_2, omega_s, omega_c):
    direction, corr, p_value = calculate_correlation_significance(
        measure_1, measure_2)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "parameter": [direction, corr],
        "type": "Correlation",
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_correlation_significance(series1, series2):
    # Ensure both series are sorted by index
    series1 = series1.sort_index()
    series2 = series2.sort_index()

    # Compute the correlation coefficient
    corr, p_value = pearsonr(series1, series2)

    # Check if the correlation is significant
    is_significant = p_value < 0.05

    # Determine the correlation direction
    if is_significant:
        direction = "Same" if corr > 0 else "Opposite"
    else:
        direction = "No significant correlation"

    return direction, corr, p_value


def get_cross_measure_correlation(df, subspace_df, subspace, breakdown_dim, measure_1, measure_2, omega_s, omega_c):
    direction, equation, p_value = calculate_cross_measure_correlation_significance(
        measure_1, measure_2)
    score_significance = 1-p_value
    score, score_impact = calculate_final_score(
        df, subspace_df, score_significance, omega_s, omega_c)

    item = {
        "subspace": subspace,
        "breakdown_dim": breakdown_dim,
        "parameter": [direction, equation],
        "type": "Cross-measure Correlation",
        "score_impact": score_impact,
        "score_significance": score_significance,
        "score": score
    }
    return item


def calculate_cross_measure_correlation_significance(series1, series2):
    # Ensure the series is sorted by index
    series1 = series1.sort_index()
    series2 = series2.sort_index()

    # Add a constant (for intercept term)
    series1_with_const = sm.add_constant(series1)

    # Fit a simple linear regression model
    model = sm.OLS(series2, series1_with_const)
    results = model.fit()

    # Get the estimated parameters and p-values
    params = results.params
    p_values = results.pvalues

    # Get the series1's name for retrieving its parameter p-value
    series1_name = series1.name if series1.name else 'x1'

    # Check the significance of the slope coefficient (measure1)
    is_significant = p_values[series1_name] < 0.05

    # Determine the correlation direction
    if is_significant:
        direction = "Same" if params[series1_name] > 0 else "Opposite"
        b0 = params['const']
        b1 = params[series1_name]
        equation = f"measure2 = {b0} + {b1}*measure1"
    else:
        direction = "No significant correlation"
        equation = None

    return direction, equation, p_values[series1_name]
