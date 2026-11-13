import os
import csv
import math
import numpy as np

def run_part1():
    # Load data
    data = []
    with open("data/records.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([
                row[0],
                float(row[1]), float(row[2]), float(row[3]),
                float(row[4]), float(row[5]), int(row[6]),
                int(row[7]), int(row[8]), float(row[9])
            ])

    n = len(data)
    values = np.array([r[9] for r in data])
    
    # Task 1.1: Estimate with an interval
    mean_value = np.mean(values)
    sd_value = np.std(values, ddof=1) # sample SD
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: Test a planning claim (H0: mu = 8.0)
    mu_hypo = 8.0
    z_stat = (mean_value - mu_hypo) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # Task 1.3: Size a future sample (E = 0.5)
    E = 0.5
    n_required = math.ceil((1.96 * sd_value / E) ** 2)

    # Task 1.4: Simple regression by closed form (value on X4: expected_salary)
    x4 = np.array([r[4] for r in data])
    cov_x4_val = np.cov(x4, values, ddof=1)[0, 1]
    var_x4 = np.var(x4, ddof=1)
    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_value - reg_slope * np.mean(x4)
    
    fitted_val = reg_slope * x4 + reg_intercept
    tss = np.sum((values - mean_value) ** 2)
    ess = np.sum((fitted_val - mean_value) ** 2)
    rss = np.sum((values - fitted_val) ** 2)
    tss_equals_ess_plus_rss = "yes" if abs(tss - (ess + rss)) < 1e-6 * tss else "no"
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = np.mean(np.abs(values - fitted_val))

    # Task 1.5: Fit the same line by gradient descent
    x4_mean = np.mean(x4)
    x4_sd = np.std(x4, ddof=1)
    z_x4 = (x4 - x4_mean) / x4_sd
    
    m_gd = 0.0
    c_gd = 0.0
    eta = 0.1
    
    def compute_cost(m, c):
        y_pred = m * z_x4 + c
        return np.mean((values - y_pred) ** 2)

    gd_j0 = compute_cost(m_gd, c_gd)
    gd_costs = []
    
    for epoch in range(1, 501):
        y_pred = m_gd * z_x4 + c_gd
        e = values - y_pred
        dm = -(2.0 / n) * np.sum(z_x4 * e)
        dc = -(2.0 / n) * np.sum(e)
        m_gd -= eta * dm
        c_gd -= eta * dc
        if epoch == 1: gd_j1 = compute_cost(m_gd, c_gd)
        if epoch == 10: gd_j10 = compute_cost(m_gd, c_gd)
        if epoch == 100: gd_j100 = compute_cost(m_gd, c_gd)
        if epoch == 500: gd_j500 = compute_cost(m_gd, c_gd)

    gd_slope = m_gd / x4_sd
    gd_intercept = c_gd - m_gd * x4_mean / x4_sd
    gd_slope_gap = abs(gd_slope - reg_slope)

    # Task 1.6: Check for a funnel, then apply log remedy
    sort_idx = np.argsort(fitted_val, kind='stable')
    sorted_residuals = (values - fitted_val)[sort_idx]
    lower_res = sorted_residuals[:400]
    upper_res = sorted_residuals[400:]
    resid_sd_lower_raw = np.std(lower_res, ddof=1)
    resid_sd_upper_raw = np.std(upper_res, ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log regression: ln(value) on ln(X4)
    log_val = np.log(values)
    log_x4 = np.log(x4)
    log_mean_x = np.mean(log_x4)
    log_mean_y = np.mean(log_val)
    log_cov = np.cov(log_x4, log_val, ddof=1)[0, 1]
    log_var = np.var(log_x4, ddof=1)
    loglog_slope = log_cov / log_var
    loglog_intercept = log_mean_y - loglog_slope * log_mean_x
    loglog_fitted = loglog_slope * log_x4 + loglog_intercept
    
    loglog_rss = np.sum((log_val - loglog_fitted) ** 2)
    loglog_tss = np.sum((log_val - log_mean_y) ** 2)
    loglog_r2 = 1.0 - loglog_rss / loglog_tss
    loglog_rmse = math.sqrt(loglog_rss / n)

    loglog_sort_idx = np.argsort(loglog_fitted, kind='stable')
    loglog_residuals = (log_val - loglog_fitted)[loglog_sort_idx]
    resid_sd_lower_log = np.std(loglog_residuals[:400], ddof=1)
    resid_sd_upper_log = np.std(loglog_residuals[400:], ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log
    
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    # Task 1.7: Multiple regression by normal equation
    # Predictors: constant (1), ln(X4), X1, X2, X3, X5, X6, X7
    x1 = np.array([r[1] for r in data])
    x2 = np.array([r[2] for r in data])
    x3 = np.array([r[3] for r in data])
    x5 = np.array([r[5] for r in data])
    x6 = np.array([r[6] for r in data])
    x7 = np.array([r[7] for r in data])

    X_mat = np.column_stack([np.ones(n), log_x4, x1, x2, x3, x5, x6, x7])
    Y_mat = log_val
    
    # Beta = (X^T X)^-1 X^T Y
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ Y_mat)
    
    mr_fitted = X_mat @ beta
    mr_rss = np.sum((Y_mat - mr_fitted) ** 2)
    mr_r2 = 1.0 - mr_rss / loglog_tss
    p_preds = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_preds - 1)

    # Save outputs/part1_results.txt
    os.makedirs("outputs", exist_ok=True)
    results = [
        ("n", f"{n}"),
        ("mean_value", f"{mean_value:.4f}"),
        ("sd_value", f"{sd_value:.4f}"),
        ("se_value", f"{se_value:.4f}"),
        ("ci95_low", f"{ci95_low:.4f}"),
        ("ci95_high", f"{ci95_high:.4f}"),
        ("z_stat", f"{z_stat:.4f}"),
        ("reject_h0", f"{reject_h0}"),
        ("n_required", f"{n_required}"),
        ("reg_slope", f"{reg_slope:.4f}"),
        ("reg_intercept", f"{reg_intercept:.4f}"),
        ("tss", f"{tss:.4f}"),
        ("ess", f"{ess:.4f}"),
        ("rss", f"{rss:.4f}"),
        ("tss_equals_ess_plus_rss", f"{tss_equals_ess_plus_rss}"),
        ("r2", f"{r2:.4f}"),
        ("rmse", f"{rmse:.4f}"),
        ("mae", f"{mae:.4f}"),
        ("gd_j0", f"{gd_j0:.4f}"),
        ("gd_j1", f"{gd_j1:.4f}"),
        ("gd_j10", f"{gd_j10:.4f}"),
        ("gd_j100", f"{gd_j100:.4f}"),
        ("gd_j500", f"{gd_j500:.4f}"),
        ("gd_slope", f"{gd_slope:.4f}"),
        ("gd_intercept", f"{gd_intercept:.4f}"),
        ("gd_slope_gap", f"{gd_slope_gap:.6f}"),
        ("resid_sd_lower_raw", f"{resid_sd_lower_raw:.4f}"),
        ("resid_sd_upper_raw", f"{resid_sd_upper_raw:.4f}"),
        ("resid_sd_ratio_raw", f"{resid_sd_ratio_raw:.4f}"),
        ("loglog_intercept", f"{loglog_intercept:.4f}"),
        ("loglog_slope", f"{loglog_slope:.4f}"),
        ("loglog_r2", f"{loglog_r2:.4f}"),
        ("loglog_rmse", f"{loglog_rmse:.4f}"),
        ("resid_sd_lower_log", f"{resid_sd_lower_log:.4f}"),
        ("resid_sd_upper_log", f"{resid_sd_upper_log:.4f}"),
        ("resid_sd_ratio_log", f"{resid_sd_ratio_log:.4f}"),
        ("mr_beta_0", f"{beta[0]:.4f}"),
        ("mr_beta_1", f"{beta[1]:.4f}"),
        ("mr_beta_2", f"{beta[2]:.4f}"),
        ("mr_beta_3", f"{beta[3]:.4f}"),
        ("mr_beta_4", f"{beta[4]:.4f}"),
        ("mr_beta_5", f"{beta[5]:.4f}"),
        ("mr_beta_6", f"{beta[6]:.4f}"),
        ("mr_beta_7", f"{beta[7]:.4f}"),
        ("mr_r2", f"{mr_r2:.4f}"),
        ("mr_adj_r2", f"{mr_adj_r2:.4f}"),
        ("loglog_adj_r2", f"{loglog_adj_r2:.4f}")
    ]

    with open("outputs/part1_results.txt", "w") as f:
        for key, val in results:
            f.write(f"{key} {val}\n")
            
    print("Part 1 completed successfully.")

if __name__ == "__main__":
    run_part1()