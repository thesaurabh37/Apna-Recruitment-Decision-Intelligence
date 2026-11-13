import os
import csv
import math
import numpy as np

def run_part2():
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

    # Task 2.1: Stratified 70/30 split
    pos_rows = [r for r in data if r[8] == 1]
    neg_rows = [r for r in data if r[8] == 0]

    n_pos = len(pos_rows) # 170
    n_neg = len(neg_rows) # 630

    train_pos_count = math.floor(0.7 * n_pos) # 119
    test_pos_count = n_pos - train_pos_count # 51

    train_neg_count = math.floor(0.7 * n_neg) # 441
    test_neg_count = n_neg - train_neg_count # 189

    # Keep file order for each outcome group
    train_data = neg_rows[:train_neg_count] + pos_rows[:train_pos_count]
    test_data = neg_rows[train_neg_count:] + pos_rows[train_pos_count:]

    # Re-sort train and test to preserve original relative file order
    # Since neg_rows and pos_rows were already in file order, let's sort by original index or just filter by original file order.
    # Actually, simpler: iterate through all data and put into train/test based on counts to preserve exact file order.

    train_indices_neg = set(tuple(r) for r in neg_rows[:train_neg_count])
    train_indices_pos = set(tuple(r) for r in pos_rows[:train_pos_count])

    train_set = [r for r in data if (tuple(r) in train_indices_neg or tuple(r) in train_indices_pos)]
    test_set = [r for r in data if r not in train_set]

    train_rows = len(train_set)
    train_positives = sum(1 for r in train_set if r[8] == 1)
    test_rows = len(test_set)
    test_positives = sum(1 for r in test_set if r[8] == 1)

    # Extract features X1 to X7 (indices 1 to 7) and outcome (index 8)
    def extract_features_labels(dataset):
        X = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in dataset])
        y = np.array([r[8] for r in dataset])
        return X, y

    X_train_raw, y_train = extract_features_labels(train_set)
    X_test_raw, y_test = extract_features_labels(test_set)

    # Standardise features using train mean and sample SD
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    def sigmoid(z):
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_bce(X, y, weights, bias):
        z = X @ weights + bias
        p = sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Task 2.2: Logistic regression from scratch
    n_tr, n_feats = X_train.shape
    weights = np.zeros(n_feats)
    bias = 0.0
    eta = 0.1

    loss_epoch_0 = compute_bce(X_train, y_train, weights, bias)

    for epoch in range(1, 1001):
        z = X_train @ weights + bias
        preds = sigmoid(z)
        error = preds - y_train
        dw = (1.0 / n_tr) * (X_train.T @ error)
        db = (1.0 / n_tr) * np.sum(error)

        weights -= eta * dw
        bias -= eta * db

        if epoch == 1: loss_epoch_1 = compute_bce(X_train, y_train, weights, bias)
        if epoch == 10: loss_epoch_10 = compute_bce(X_train, y_train, weights, bias)
        if epoch == 100: loss_epoch_100 = compute_bce(X_train, y_train, weights, bias)
        if epoch == 1000: loss_epoch_1000 = compute_bce(X_train, y_train, weights, bias)

    test_bce = compute_bce(X_test, y_test, weights, bias)

    # Task 2.3: Learning-rate effect (200 updates for eta in {0.01, 0.1, 1.0})
    def run_lr_test(lr):
        w = np.zeros(n_feats)
        b = 0.0
        for _ in range(200):
            z = X_train @ w + b
            preds = sigmoid(z)
            error = preds - y_train
            dw = (1.0 / n_tr) * (X_train.T @ error)
            db = (1.0 / n_tr) * np.sum(error)
            w -= lr * dw
            b -= lr * db
        return compute_bce(X_train, y_train, w, b)

    loss_epoch_200_eta_0_01 = run_lr_test(0.01)
    loss_epoch_200_eta_0_1 = run_lr_test(0.1)
    loss_epoch_200_eta_1_0 = run_lr_test(1.0)

    # Task 2.4: Odds ratios
    odds_ratios = np.exp(weights)

    # Task 2.5: Evaluate test rows at threshold 0.5 and 0.3
    test_probs = sigmoid(X_test @ weights + bias)

    def evaluate_metrics(probs, y_true, tau):
        preds = (probs >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return tp, fp, tn, fn, accuracy, precision, recall, f1, specificity

    t05_tp, t05_fp, t05_tn, t05_fn, t05_acc, t05_prec, t05_rec, t05_f1, t05_spec = evaluate_metrics(test_probs, y_test, 0.5)
    t03_tp, t03_fp, t03_tn, t03_fn, t03_acc, t03_prec, t03_rec, t03_f1, t03_spec = evaluate_metrics(test_probs, y_test, 0.3)

    blind_accuracy = np.sum(y_test == 0) / len(y_test)

    # Task 2.6: Threshold sweep for tau in {0.1, 0.2, ..., 0.9}
    thresholds_rows = []
    tau_vals = [round(x * 0.1, 1) for x in range(1, 10)]
    for tau in tau_vals:
        preds = (test_probs >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (y_test == 1)))
        fp = int(np.sum((preds == 1) & (y_test == 0)))
        tn = int(np.sum((preds == 0) & (y_test == 0)))
        fn = int(np.sum((preds == 0) & (y_test == 1)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        thresholds_rows.append([tau, tp, fp, tn, fn, tpr, fpr])

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part2_thresholds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        for row in thresholds_rows:
            writer.writerow([f"{row[0]:.1f}", row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}"])

    # Task 2.7: K-Nearest Neighbours with 5-fold CV
    # K values: 1, 3, 5, ..., 21
    k_list = list(range(1, 22, 2))
    knn_cv_results = []

    # 5-fold CV based on 0-based position in training set modulo 5
    folds = [i % 5 for i in range(n_tr)]

    best_k = 1
    best_mean_f1 = -1.0

    for k in k_list:
        fold_f1s = []
        for f_idx in range(5):
            val_mask = np.array([f == f_idx for f in folds])
            train_mask = ~val_mask

            X_fold_tr = X_train[train_mask]
            y_fold_tr = y_train[train_mask]
            X_fold_val = X_train[val_mask]
            y_fold_val = y_train[val_mask]

            # Predict validation fold
            val_preds = []
            for query in X_fold_val:
                # Euclidean distance
                dists = np.sqrt(np.sum((X_fold_tr - query) ** 2, axis=1))
                # Sort by distance, then position in file order (which is index order in X_fold_tr)
                sorted_indices = np.lexsort((np.arange(len(dists)), dists))
                k_indices = sorted_indices[:k]
                k_labels = y_fold_tr[k_indices]

                pos_count = np.sum(k_labels == 1)
                pred = 1 if pos_count >= (k + 1) / 2 else 0
                val_preds.append(pred)

            val_preds = np.array(val_preds)
            tp = np.sum((val_preds == 1) & (y_fold_val == 1))
            fp = np.sum((val_preds == 1) & (y_fold_val == 0))
            fn = np.sum((val_preds == 0) & (y_fold_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            fold_f1s.append(f1)

        mean_cv_f1 = np.mean(fold_f1s)
        knn_cv_results.append([k, mean_cv_f1])
        if mean_cv_f1 > best_mean_f1:
            best_mean_f1 = mean_cv_f1
            best_k = k

    with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for row in knn_cv_results:
            writer.writerow([row[0], f"{row[1]:.4f}"])

    # Refit KNN with all 560 training rows as reference and evaluate on test set using best_k
    knn_test_preds = []
    for query in X_test:
        dists = np.sqrt(np.sum((X_train - query) ** 2, axis=1))
        sorted_indices = np.lexsort((np.arange(len(dists)), dists))
        k_indices = sorted_indices[:best_k]
        k_labels = y_train[k_indices]
        pos_count = np.sum(k_labels == 1)
        pred = 1 if pos_count >= (best_k + 1) / 2 else 0
        knn_test_preds.append(pred)

    knn_tp, knn_fp, knn_tn, knn_fn, knn_acc, knn_prec, knn_rec, knn_f1, _ = evaluate_metrics(np.array(knn_test_preds), y_test, 0.5)

    # Task 2.8: Choose alert threshold with money
    # Cost = 50 * FP + 400 * FN (Wait, prompt says: "each false alarm costs INR 50 and each missed positive costs INR 400, so cost = 50 * ED + 400 * ED" -> typo in brief text means FP and FN)
    # Let's check prompt wording: "each false alarm costs INR 50 and each missed positive costs INR 400, so cost = 50 * FP + 400 * FN"
    train_probs = sigmoid(X_train @ weights + bias)

    best_tau_cost = float('inf')
    best_tau = 0.05

    for k_val in range(1, 20):
        tau = k_val / 20.0
        preds = (train_probs >= tau).astype(int)
        fp = np.sum((preds == 1) & (y_train == 0))
        fn = np.sum((preds == 0) & (y_train == 1))
        cost = 50 * fp + 400 * fn
        if cost < best_tau_cost:
            best_tau_cost = cost
            best_tau = tau

    cost_tau = best_tau
    cost_train_at_tau = best_tau_cost

    # Evaluate cost on test set at cost_tau, tau=0.5, and flag nobody
    def compute_cost(y_true, preds):
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        return 50 * fp + 400 * fn

    test_preds_tau = (test_probs >= cost_tau).astype(int)
    cost_test_at_tau = compute_cost(y_test, test_preds_tau)

    test_preds_05 = (test_probs >= 0.5).astype(int)
    cost_test_at_05 = compute_cost(y_test, test_preds_05)

    test_preds_nobody = np.zeros_like(y_test)
    cost_test_flag_nobody = compute_cost(y_test, test_preds_nobody)

    # Save outputs/part2_results.txt
    results = [
        ("train_rows", f"{train_rows}"),
        ("train_positives", f"{train_positives}"),
        ("test_rows", f"{test_rows}"),
        ("test_positives", f"{test_positives}"),
        ("loss_epoch_0", f"{loss_epoch_0:.4f}"),
        ("loss_epoch_1", f"{loss_epoch_1:.4f}"),
        ("loss_epoch_10", f"{loss_epoch_10:.4f}"),
        ("loss_epoch_100", f"{loss_epoch_100:.4f}"),
        ("loss_epoch_1000", f"{loss_epoch_1000:.4f}"),
        ("weight_X1", f"{weights[0]:.4f}"),
        ("weight_X2", f"{weights[1]:.4f}"),
        ("weight_X3", f"{weights[2]:.4f}"),
        ("weight_X4", f"{weights[3]:.4f}"),
        ("weight_X5", f"{weights[4]:.4f}"),
        ("weight_X6", f"{weights[5]:.4f}"),
        ("weight_X7", f"{weights[6]:.4f}"),
        ("bias", f"{bias:.4f}"),
        ("test_bce", f"{test_bce:.4f}"),
        ("loss_epoch_200_eta_0.01", f"{loss_epoch_200_eta_0_01:.4f}"),
        ("loss_epoch_200_eta_0.1", f"{loss_epoch_200_eta_0_1:.4f}"),
        ("loss_epoch_200_eta_1.0", f"{loss_epoch_200_eta_1_0:.4f}"),
        ("odds_ratio_X1", f"{odds_ratios[0]:.4f}"),
        ("odds_ratio_X2", f"{odds_ratios[1]:.4f}"),
        ("odds_ratio_X3", f"{odds_ratios[2]:.4f}"),
        ("odds_ratio_X4", f"{odds_ratios[3]:.4f}"),
        ("odds_ratio_X5", f"{odds_ratios[4]:.4f}"),
        ("odds_ratio_X6", f"{odds_ratios[5]:.4f}"),
        ("odds_ratio_X7", f"{odds_ratios[6]:.4f}"),
        ("tau_0.5_tp", f"{t05_tp}"),
        ("tau_0.5_fp", f"{t05_fp}"),
        ("tau_0.5_tn", f"{t05_tn}"),
        ("tau_0.5_fn", f"{t05_fn}"),
        ("tau_0.5_accuracy", f"{t05_acc:.4f}"),
        ("tau_0.5_precision", f"{t05_prec:.4f}"),
        ("tau_0.5_recall", f"{t05_rec:.4f}"),
        ("tau_0.5_f1", f"{t05_f1:.4f}"),
        ("tau_0.5_specificity", f"{t05_spec:.4f}"),
        ("tau_0.3_tp", f"{t03_tp}"),
        ("tau_0.3_fp", f"{t03_fp}"),
        ("tau_0.3_tn", f"{t03_tn}"),
        ("tau_0.3_fn", f"{t03_fn}"),
        ("tau_0.3_accuracy", f"{t03_acc:.4f}"),
        ("tau_0.3_precision", f"{t03_prec:.4f}"),
        ("tau_0.3_recall", f"{t03_rec:.4f}"),
        ("tau_0.3_f1", f"{t03_f1:.4f}"),
        ("tau_0.3_specificity", f"{t03_spec:.4f}"),
        ("blind_accuracy", f"{blind_accuracy:.4f}"),
        ("knn_best_k", f"{best_k}"),
        ("knn_tp", f"{knn_tp}"),
        ("knn_fp", f"{knn_fp}"),
        ("knn_tn", f"{knn_tn}"),
        ("knn_fn", f"{knn_fn}"),
        ("knn_accuracy", f"{knn_acc:.4f}"),
        ("knn_precision", f"{knn_prec:.4f}"),
        ("knn_recall", f"{knn_rec:.4f}"),
        ("knn_f1", f"{knn_f1:.4f}"),
        ("cost_tau", f"{cost_tau:.2f}"),
        ("cost_train_at_tau", f"{cost_train_at_tau}"),
        ("cost_test_at_tau", f"{cost_test_at_tau}"),
        ("cost_test_at_0.5", f"{cost_test_at_05}"),
        ("cost_test_flag_nobody", f"{cost_test_flag_nobody}")
    ]

    with open("outputs/part2_results.txt", "w") as f:
        for key, val in results:
            f.write(f"{key} {val}\n")

    print("Part 2 completed successfully.")

if __name__ == "__main__":
    run_part2()