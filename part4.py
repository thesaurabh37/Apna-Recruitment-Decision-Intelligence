import os
import csv
import math
import numpy as np

def run_part4():
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

    # Re-create Task 2.1 stratified train/test split to ensure exact match
    pos_rows = [r for r in data if r[8] == 1]
    neg_rows = [r for r in data if r[8] == 0]

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)

    train_pos_count = math.floor(0.7 * n_pos)
    train_neg_count = math.floor(0.7 * n_neg)

    train_indices_neg = set(tuple(r) for r in neg_rows[:train_neg_count])
    train_indices_pos = set(tuple(r) for r in pos_rows[:train_pos_count])

    train_set = [r for r in data if (tuple(r) in train_indices_neg or tuple(r) in train_indices_pos)]
    test_set = [r for r in data if r not in train_set]

    def extract_features_labels(dataset):
        X = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in dataset])
        y = np.array([r[8] for r in dataset])
        return X, y

    X_train_raw, y_train = extract_features_labels(train_set)
    X_test_raw, y_test = extract_features_labels(test_set)

    # Standardise using train mean and sample SD
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    n_tr = len(train_set)

    # Task 4.1: Which split is purest?
    # Candidate splits on 560 training rows in original units:
    # S1: X6 = 1 (competing_offer_flag, index 6)
    # S2: X7 = 1 (referred_flag, index 7)
    # S3: X3 > 0.5 (salary_gap_ratio, index 3)
    # S4: X1 <= 24 (experience_months, index 1)

    def calc_entropy(pos_count, total_count):
        if total_count == 0 or pos_count == 0 or pos_count == total_count:
            return 0.0
        p = pos_count / total_count
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def calc_gini(pos_count, total_count):
        if total_count == 0:
            return 0.0
        p = pos_count / total_count
        return 1.0 - (p**2 + (1 - p)**2)

    root_pos = np.sum(y_train == 1)
    root_entropy = calc_entropy(root_pos, n_tr)
    root_gini = calc_gini(root_pos, n_tr)

    splits_eval = []
    
    # S1: X6 == 1
    s1_yes = [r for r in train_set if r[6] == 1]
    s1_no = [r for r in train_set if r[6] == 0]
    
    # S2: X7 == 1
    s2_yes = [r for r in train_set if r[7] == 1]
    s2_no = [r for r in train_set if r[7] == 0]

    # S3: X3 > 0.5
    s3_yes = [r for r in train_set if r[3] > 0.5]
    s3_no = [r for r in train_set if r[3] <= 0.5]

    # S4: X1 <= 24
    s4_yes = [r for r in train_set if r[1] <= 24]
    s4_no = [r for r in train_set if r[1] > 24]

    candidate_splits = [
        ("S1", s1_yes, s1_no),
        ("S2", s2_yes, s2_no),
        ("S3", s3_yes, s3_no),
        ("S4", s4_yes, s4_no)
    ]

    split_csv_rows = []
    best_info_gain = -1.0
    best_gini_gain = -1.0
    best_split_info = "S1"
    best_split_gini = "S1"

    for s_name, yes_subset, no_subset in candidate_splits:
        n_y = len(yes_subset)
        pos_y = sum(1 for r in yes_subset if r[8] == 1)
        n_n = len(no_subset)
        pos_n = sum(1 for r in no_subset if r[8] == 1)

        ent_y = calc_entropy(pos_y, n_y)
        ent_n = calc_entropy(pos_n, n_n)
        child_ent_weighted = (n_y / n_tr) * ent_y + (n_n / n_tr) * ent_n
        info_gain = root_entropy - child_ent_weighted

        gini_y = calc_gini(pos_y, n_y)
        gini_n = calc_gini(pos_n, n_n)
        child_gini_weighted = (n_y / n_tr) * gini_y + (n_n / n_tr) * gini_n
        gini_gain = root_gini - child_gini_weighted

        split_csv_rows.append([s_name, n_y, pos_y, n_n, pos_n, info_gain, gini_gain])

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split_info = s_name

        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_split_gini = s_name

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part4_splits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
        for row in split_csv_rows:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}"])

    # Task 4.2: A one-question model and two ways to combine models
    # The stump is the best split by information gain (S3: X3 > 0.5)
    stump_yes_subset = s3_yes
    stump_no_subset = s3_no
    stump_rate_yes = sum(1 for r in stump_yes_subset if r[8] == 1) / len(stump_yes_subset)
    stump_rate_no = sum(1 for r in stump_no_subset if r[8] == 1) / len(stump_no_subset)

    # Train logistic regression parameters from scratch (same as Part 2) to build models
    n_feats = X_train.shape[1]
    weights = np.zeros(n_feats)
    bias = 0.0
    eta = 0.1
    for _ in range(1000):
        z = X_train @ weights + bias
        preds_sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = preds_sig - y_train
        weights -= eta * (1.0 / n_tr) * (X_train.T @ error)
        bias -= eta * (1.0 / n_tr) * np.sum(error)

    test_probs = 1.0 / (1.0 + np.exp(-np.clip(X_test @ weights + bias, -500, 500)))

    # KNN best K from training (let's recompute or use K=15 from typical reference or compute via same logic)
    # Let's run quick CV to find knn_best_k exactly
    folds = [i % 5 for i in range(n_tr)]
    best_k = 1
    best_mean_f1 = -1.0
    for k in range(1, 22, 2):
        fold_f1s = []
        for f_idx in range(5):
            val_mask = np.array([f == f_idx for f in folds])
            train_mask = ~val_mask
            X_f_tr, y_f_tr = X_train[train_mask], y_train[train_mask]
            X_f_val, y_f_val = X_train[val_mask], y_train[val_mask]
            val_preds = []
            for query in X_f_val:
                dists = np.sqrt(np.sum((X_f_tr - query) ** 2, axis=1))
                sorted_idx = np.lexsort((np.arange(len(dists)), dists))
                k_labels = y_f_tr[sorted_idx[:k]]
                val_preds.append(1 if np.sum(k_labels == 1) >= (k + 1) / 2 else 0)
            val_preds = np.array(val_preds)
            tp = np.sum((val_preds == 1) & (y_f_val == 1))
            fp = np.sum((val_preds == 1) & (y_f_val == 0))
            fn = np.sum((val_preds == 0) & (y_f_val == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fold_f1s.append((2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0)
        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_k = k

    # KNN test predictions and probability (share of positive neighbours)
    knn_test_preds = []
    knn_test_probs = []
    for query in X_test:
        dists = np.sqrt(np.sum((X_train - query) ** 2, axis=1))
        sorted_idx = np.lexsort((np.arange(len(dists)), dists))
        k_labels = y_train[sorted_idx[:best_k]]
        pos_share = np.sum(k_labels == 1) / best_k
        knn_test_probs.append(pos_share)
        knn_test_preds.append(1 if pos_share >= 0.5 else 0)
    knn_test_probs = np.array(knn_test_probs)

    # Stump test probabilities
    stump_test_probs = np.array([stump_rate_yes if r[3] > 0.5 else stump_rate_no for r in test_set])

    # Five models on 240 test rows:
    # 1. logistic: positive when prob >= 0.3
    log_preds = (test_probs >= 0.3).astype(int)
    # 2. knn: positive when >= (K+1)/2 neighbours positive (prob >= 0.5 effectively)
    knn_preds = np.array(knn_test_preds)
    # 3. stump: positive when prob >= 0.3
    stump_preds = (stump_test_probs >= 0.3).astype(int)
    # 4. hardvote: positive when at least two of the three above predict positive
    hard_votes = log_preds + knn_preds + stump_preds
    hard_preds = (hard_votes >= 2).astype(int)
    # 5. softvote: positive when mean probability >= 0.3
    mean_probs = (test_probs + knn_test_probs + stump_test_probs) / 3.0
    soft_preds = (mean_probs >= 0.3).astype(int)

    voting_models_eval = [
        ("logistic", log_preds),
        ("knn", knn_preds),
        ("stump", stump_preds),
        ("hardvote", hard_preds),
        ("softvote", soft_preds)
    ]

    voting_csv_rows = []
    for m_name, preds in voting_models_eval:
        tp = int(np.sum((preds == 1) & (y_test == 1)))
        fp = int(np.sum((preds == 1) & (y_test == 0)))
        tn = int(np.sum((preds == 0) & (y_test == 0)))
        fn = int(np.sum((preds == 0) & (y_test == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        voting_csv_rows.append([m_name, tp, fp, tn, fn, precision, recall, f1])

    with open("outputs/part4_voting.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        for row in voting_csv_rows:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}"])

    # Task 4.3: Two rounds of AdaBoost by hand
    y_ab = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    weights_r1 = np.ones(8) / 8.0
    
    # Round 1 error
    err_mask_r1 = (h1 != y_ab)
    round1_error = np.sum(weights_r1[err_mask_r1])
    round1_alpha = 0.5 * math.log((1.0 - round1_error) / round1_error)

    # Weights unnormalised and after round 1
    unnorm_w1 = weights_r1 * np.exp(-round1_alpha * y_ab * h1)
    weights_r2 = unnorm_w1 / np.sum(unnorm_w1)

    # Round 2 error
    err_mask_r2 = (h2 != y_ab)
    round2_error = np.sum(weights_r2[err_mask_r2])
    round2_alpha = 0.5 * math.log((1.0 - round2_error) / round2_error)

    unnorm_w2 = weights_r2 * np.exp(-round2_alpha * y_ab * h2)
    weights_r3 = unnorm_w2 / np.sum(unnorm_w2)

    adaboost_rows = []
    for idx in range(8):
        adaboost_rows.append([1, idx + 1, weights_r1[idx], unnorm_w1[idx], weights_r2[idx]])
    for idx in range(8):
        adaboost_rows.append([2, idx + 1, weights_r2[idx], unnorm_w2[idx], weights_r3[idx]])

    with open("outputs/part4_adaboost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight before", "weight unnormalised", "weight after"])
        for row in adaboost_rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    ensemble_preds = np.sign(round1_alpha * h1 + round2_alpha * h2)
    ensemble_accuracy = np.sum(ensemble_preds == y_ab) / 8.0

    # Task 4.4: Out-of-bag share
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560.0

    # Task 4.5: Size the neural networks a later project would need
    # Model A: Image input 28x28x1, 10 classes
    # Option 1: FC 784 -> 16 -> 16 -> 10
    # Option 2: CNN layers and parameters
    # Layer 1: conv1 (5x5 filters, 6 filters, stride 1, padding 2) -> output shape 28x28x6. Parameters: (5*5*1 + 1) * 6 = 156
    # ReLU & MaxPool 2x2 stride 2 -> output shape 14x14x6.
    # Layer 2: conv2 (5x5 filters, 16 filters, stride 1, padding 0) -> output shape (14-5+0)/1 + 1 = 10x10x16. Parameters: (5*5*6 + 1) * 16 = 2416
    # ReLU & MaxPool 2x2 stride 2 -> output shape 5x5x16 = 400.
    # Layer 3: dense1 (32 neurons). Parameters: 400 * 32 + 32 = 12832
    # Layer 4: dense2 (output 10 neurons). Parameters: 32 * 10 + 10 = 330

    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    cnn_dense1 = 400 * 32 + 32
    cnn_dense2 = 32 * 10 + 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    architecture_rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2]
    ]

    with open("outputs/part4_architecture.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        for row in architecture_rows:
            writer.writerow([row[0], row[1]])

    # Additional questions in Part 4: fc_layer_params, conv_layer_params, rnn_total
    # fc layer 128x128x3 input into 1,024 neurons: (128*128*3 + 1) * 1024 = 50,341,888
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    # conv layer same 128x128x3 input with 64 filters of size 3x3 (stride 1, padding 1 assumed standard): (3*3*3 + 1) * 64 = 1,792
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    # Model B: Token-sequence input (RNN)
    # 64-dim word vector Xt, 128-dim hidden state, 16 class scores
    # Wxh: 128 x 64 = 8,192
    # Whh: 128 x 128 = 16,384
    # Why: 16 x 128 = 2,048
    # bh: 128
    # by: 16
    wxh_p = 128 * 64
    whh_p = 128 * 128
    why_p = 16 * 128
    bh_p = 128
    by_p = 16
    rnn_total = wxh_p + whh_p + why_p + bh_p + by_p

    # Save outputs/part4_results.txt
    results = [
        ("root_entropy", f"{root_entropy:.4f}"),
        ("root_gini", f"{root_gini:.4f}"),
        ("best_split_by_info_gain", f"{best_split_info}"),
        ("best_split_by_gini_gain", f"{best_split_gini}"),
        ("stump_rate_yes", f"{stump_rate_yes:.4f}"),
        ("stump_rate_no", f"{stump_rate_no:.4f}"),
        ("round1_error", f"{round1_error:.4f}"),
        ("round1_alpha", f"{round1_alpha:.4f}"),
        ("round2_error", f"{round2_error:.4f}"),
        ("round2_alpha", f"{round2_alpha:.4f}"),
        ("ensemble_accuracy", f"{ensemble_accuracy:.4f}"),
        ("oob_fraction", f"{oob_fraction:.4f}"),
        ("mlp_total", f"{mlp_total}"),
        ("cnn_total", f"{cnn_total}"),
        ("fc_layer_params", f"{fc_layer_params}"),
        ("conv_layer_params", f"{conv_layer_params}"),
        ("rnn_total", f"{rnn_total}")
    ]

    with open("outputs/part4_results.txt", "w") as f:
        for key, val in results:
            f.write(f"{key} {val}\n")

    print("Part 4 completed successfully.")

if __name__ == "__main__":
    run_part4()