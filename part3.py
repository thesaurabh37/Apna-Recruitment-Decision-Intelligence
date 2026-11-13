import os
import csv
import math
import numpy as np

def run_part3():
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
    # Continuous predictors X1 to X5 (indices 1 to 5)
    X_raw = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in data])
    
    mean_X = np.mean(X_raw, axis=0)
    sd_X = np.std(X_raw, axis=0, ddof=1)
    
    # Standardise X1 to X5 over all 800 rows
    X_std = (X_raw - mean_X) / sd_X

    # Task 3.1: K-means from scratch for K = 2, 3, 4, 5, 6
    wcss_results = {}
    cluster_assignments_all = {}

    for k in range(2, 7):
        # Pinned starting centroids: rows with 0-based indices 0, floor(800/K), 2*floor(800/K), ..., (K-1)*floor(800/K)
        step = n // k
        centroid_indices = [i * step for i in range(k)]
        centroids = X_std[centroid_indices].copy()
        
        assignments = np.zeros(n, dtype=int)
        
        while True:
            # Assign each row to nearest centroid by squared Euclidean distance (ties go to lower-numbered centroid)
            # D[i, j] = ||X[i] - C[j]||^2
            dists = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            new_assignments = np.argmin(dists, axis=1) # argmin naturally picks lower-numbered index on ties
            
            if np.all(new_assignments == assignments):
                break
            assignments = new_assignments
            
            # Recompute centroids
            for c_idx in range(k):
                members = X_std[assignments == c_idx]
                if len(members) > 0:
                    centroids[c_idx] = np.mean(members, axis=0)
                # centroid with no members stays where it is

        cluster_assignments_all[k] = assignments
        
        # Calculate WCSS
        wcss = 0.0
        for i in range(n):
            c_idx = assignments[i]
            wcss += np.sum((X_std[i] - centroids[c_idx]) ** 2)
        wcss_results[k] = wcss

    wcss_k2 = wcss_results[2]
    wcss_k3 = wcss_results[3]
    wcss_k4 = wcss_results[4]
    wcss_k5 = wcss_results[5]
    wcss_k6 = wcss_results[6]

    # Task 3.2: Choose K with the elbow
    # smallest K in {3, 4, 5} for which WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = 3
    for k in [3, 4, 5]:
        drop_current = wcss_results[k] - wcss_results[k+1]
        drop_prev = wcss_results[k-1] - wcss_results[k]
        if drop_current < 0.5 * drop_prev:
            elbow_k = k
            break

    # Use K=3 for the rest of this Part
    k3_assignments = cluster_assignments_all[3]

    # Task 3.3: Profile the K=3 segments
    # Order the three clusters by ascending mean of X3 (in original units) and call them segments A, B and C.
    cluster_indices = [0, 1, 2]
    cluster_x3_means = []
    for c_idx in cluster_indices:
        mask = (k3_assignments == c_idx)
        mean_x3_val = np.mean([data[i][3] for i in range(n) if mask[i]])
        cluster_x3_means.append((c_idx, mean_x3_val))
    
    cluster_x3_means.sort(key=lambda x: x[1])
    
    segment_names = ['A', 'B', 'C']
    c_map = {cluster_x3_means[i][0]: segment_names[i] for i in range(3)}

    segments_data = []
    for orig_c_idx, seg_name in sorted(c_map.items(), key=lambda x: x[1]):
        mask = (k3_assignments == orig_c_idx)
        seg_rows = [data[i] for i in range(n) if mask[i]]
        size_seg = len(seg_rows)
        
        mean_x1 = np.mean([r[1] for r in seg_rows])
        mean_x2 = np.mean([r[2] for r in seg_rows])
        mean_x3 = np.mean([r[3] for r in seg_rows])
        mean_x4 = np.mean([r[4] for r in seg_rows])
        mean_x5 = np.mean([r[5] for r in seg_rows])
        outcome_rate = np.mean([r[8] for r in seg_rows])
        mean_val = np.mean([r[9] for r in seg_rows])
        
        segments_data.append({
            'segment': seg_name,
            'size': size_seg,
            'mean_X1': mean_x1,
            'mean_X2': mean_x2,
            'mean_X3': mean_x3,
            'mean_X4': mean_x4,
            'mean_X5': mean_x5,
            'outcome_rate': outcome_rate,
            'mean_value': mean_val
        })

    size_segment_A = segments_data[0]['size']
    size_segment_B = segments_data[1]['size']
    size_segment_C = segments_data[2]['size']

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part3_segments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
        for s in segments_data:
            writer.writerow([
                s['segment'], s['size'],
                f"{s['mean_X1']:.4f}", f"{s['mean_X2']:.4f}", f"{s['mean_X3']:.4f}",
                f"{s['mean_X4']:.4f}", f"{s['mean_X5']:.4f}",
                f"{s['outcome_rate']:.4f}", f"{s['mean_value']:.4f}"
            ])

    # Task 3.4: Hierarchical clustering on supplied table
    dissimilarity_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])
    cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]

    def run_hierarchical(linkage_type):
        clusters = [[name] for name in cohort_names]
        dist_mat = dissimilarity_matrix.copy()
        merges = []
        
        for step in range(1, 6):
            min_dist = float('inf')
            pair = (0, 1)
            
            n_c = len(clusters)
            for i in range(n_c):
                for j in range(i + 1, n_c):
                    # compute linkage distance between clusters[i] and clusters[j]
                    ci = clusters[i]
                    cj = clusters[j]
                    
                    indices_i = [cohort_names.index(member) for member in ci]
                    indices_j = [cohort_names.index(member) for member in cj]
                    
                    sub_dists = dissimilarity_matrix[np.ix_(indices_i, indices_j)]
                    
                    if linkage_type == 'single':
                        d = np.min(sub_dists)
                    elif linkage_type == 'complete':
                        d = np.max(sub_dists)
                    elif linkage_type == 'average':
                        d = np.mean(sub_dists)
                        
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)
                        
            i, j = sorted(pair)
            ci = clusters[i]
            cj = clusters[j]
            
            # cluster_a is the cluster whose smallest member number is the smaller one
            # members are named like E1, E2... extract numeric part for comparison
            def get_min_member_num(cluster_members):
                nums = [int(m[1:]) for m in cluster_members]
                return min(nums)
                
            num_i = get_min_member_num(ci)
            num_j = get_min_member_num(cj)
            
            if num_i < num_j:
                c_a, c_b = ci, cj
            else:
                c_a, c_b = cj, ci
                
            # Write as member names in ascending order joined by +
            c_a_str = "+".join(sorted(c_a, key=lambda x: int(x[1:])))
            c_b_str = "+".join(sorted(c_b, key=lambda x: int(x[1:])))
            
            merged_cluster = ci + cj
            
            merges.append({
                'linkage': linkage_type,
                'merge_no': step,
                'cluster_a': c_a_str,
                'cluster_b': c_b_str,
                'height': min_dist
            })
            
            # update clusters list
            new_clusters = [clusters[idx] for idx in range(n_c) if idx != i and idx != j]
            new_clusters.append(merged_cluster)
            clusters = new_clusters
            
        return merges

    single_merges = run_hierarchical('single')
    complete_merges = run_hierarchical('complete')
    average_merges = run_hierarchical('average')

    all_hierarchical_rows = single_merges + complete_merges + average_merges

    with open("outputs/part3_hierarchical.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        for row in all_hierarchical_rows:
            writer.writerow([row['linkage'], row['merge_no'], row['cluster_a'], row['cluster_b'], f"{row['height']:.4f}"])

    # Cut dendrogram: four gaps between consecutive merge heights (merge 2 - merge 1, ..., merge 5 - merge 4)
    # Cut lies in the largest gap, number of clusters = 6 - number of merges below the cut
    def get_clusters_at_cut(merges):
        heights = [m['height'] for m in merges]
        gaps = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        max_gap_idx = np.argmax(gaps) # 0-based index of gap (0: m2-m1, 1: m3-m2, etc.)
        # merges below cut are from index 0 up to max_gap_idx
        merges_below = max_gap_idx + 1
        clusters_count = 6 - merges_below
        return clusters_count

    single_clusters_at_cut = get_clusters_at_cut(single_merges)
    complete_clusters_at_cut = get_clusters_at_cut(complete_merges)
    average_clusters_at_cut = get_clusters_at_cut(average_merges)

    # Task 3.5: PCA by the seven steps on standardised 800 x 5 matrix
    # (1) mean vector (all zeros); (2) centre data (already centered); (3) covariance matrix S = (1/(n-1)) * X^T X
    cov_matrix = (1.0 / (n - 1)) * (X_std.T @ X_std)
    # (4) eigenvalues and eigenvectors using numpy.linalg.eigh
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # (5) sort by eigenvalue, largest first
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Sign convention: flip each eigenvector so that its entry of largest absolute value is positive
    for col in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, col]
        max_abs_idx = np.argmax(np.abs(vec))
        if vec[max_abs_idx] < 0:
            eigenvectors[:, col] = -vec

    eigenvalue_sum = np.sum(eigenvalues)
    
    explained_ratios = eigenvalues / eigenvalue_sum
    cumulative_ratios = np.cumsum(explained_ratios)
    
    # Components for 80%: smallest number of components with cumulative ratio at least 0.8
    components_for_80pct = int(np.where(cumulative_ratios >= 0.8)[0][0] + 1)

    # (6) keep top two eigenvectors, (7) project Z = X W
    W = eigenvectors[:, :2]
    Z = X_std @ W

    pc1_loadings = eigenvectors[:, 0]
    pc2_loadings = eigenvectors[:, 1]

    # Projections of first three data rows
    row1_pc1, row1_pc2 = Z[0, 0], Z[0, 1]
    row2_pc1, row2_pc2 = Z[1, 0], Z[1, 1]
    row3_pc1, row3_pc2 = Z[2, 0], Z[2, 1]

    # Mean PC1 score of records in each Task 3.3 segment
    mean_pc1_segments = {}
    for orig_c_idx, seg_name in c_map.items():
        mask = (k3_assignments == orig_c_idx)
        seg_pc1s = Z[mask, 0]
        mean_pc1_segments[seg_name] = np.mean(seg_pc1s)

    # Save outputs/part3_results.txt
    results = [
        ("wcss_k2", f"{wcss_k2:.4f}"),
        ("wcss_k3", f"{wcss_k3:.4f}"),
        ("wcss_k4", f"{wcss_k4:.4f}"),
        ("wcss_k5", f"{wcss_k5:.4f}"),
        ("wcss_k6", f"{wcss_k6:.4f}"),
        ("elbow_k", f"{elbow_k}"),
        ("size_segment_A", f"{size_segment_A}"),
        ("size_segment_B", f"{size_segment_B}"),
        ("size_segment_C", f"{size_segment_C}"),
        ("single_clusters_at_cut", f"{single_clusters_at_cut}"),
        ("complete_clusters_at_cut", f"{complete_clusters_at_cut}"),
        ("average_clusters_at_cut", f"{average_clusters_at_cut}"),
        ("eigenvalue_sum", f"{eigenvalue_sum:.4f}"),
        ("eigenvalue_1", f"{eigenvalues[0]:.4f}"),
        ("eigenvalue_2", f"{eigenvalues[1]:.4f}"),
        ("eigenvalue_3", f"{eigenvalues[2]:.4f}"),
        ("eigenvalue_4", f"{eigenvalues[3]:.4f}"),
        ("eigenvalue_5", f"{eigenvalues[4]:.4f}"),
        ("explained_ratio_1", f"{explained_ratios[0]:.4f}"),
        ("explained_ratio_2", f"{explained_ratios[1]:.4f}"),
        ("explained_ratio_3", f"{explained_ratios[2]:.4f}"),
        ("explained_ratio_4", f"{explained_ratios[3]:.4f}"),
        ("explained_ratio_5", f"{explained_ratios[4]:.4f}"),
        ("cumulative_ratio_1", f"{cumulative_ratios[0]:.4f}"),
        ("cumulative_ratio_2", f"{cumulative_ratios[1]:.4f}"),
        ("cumulative_ratio_3", f"{cumulative_ratios[2]:.4f}"),
        ("cumulative_ratio_4", f"{cumulative_ratios[3]:.4f}"),
        ("cumulative_ratio_5", f"{cumulative_ratios[5-1]:.4f}"),
        ("components_for_80pct", f"{components_for_80pct}"),
        ("pc1_loading_X1", f"{pc1_loadings[0]:.4f}"),
        ("pc1_loading_X2", f"{pc1_loadings[1]:.4f}"),
        ("pc1_loading_X3", f"{pc1_loadings[2]:.4f}"),
        ("pc1_loading_X4", f"{pc1_loadings[3]:.4f}"),
        ("pc1_loading_X5", f"{pc1_loadings[4]:.4f}"),
        ("pc2_loading_X1", f"{pc2_loadings[0]:.4f}"),
        ("pc2_loading_X2", f"{pc2_loadings[1]:.4f}"),
        ("pc2_loading_X3", f"{pc2_loadings[2]:.4f}"),
        ("pc2_loading_X4", f"{pc2_loadings[3]:.4f}"),
        ("pc2_loading_X5", f"{pc2_loadings[4]:.4f}"),
        ("row1_pc1", f"{row1_pc1:.4f}"),
        ("row1_pc2", f"{row1_pc2:.4f}"),
        ("row2_pc1", f"{row2_pc1:.4f}"),
        ("row2_pc2", f"{row2_pc2:.4f}"),
        ("row3_pc1", f"{row3_pc1:.4f}"),
        ("row3_pc2", f"{row3_pc2:.4f}"),
        ("mean_pc1_segment_A", f"{mean_pc1_segments['A']:.4f}"),
        ("mean_pc1_segment_B", f"{mean_pc1_segments['B']:.4f}"),
        ("mean_pc1_segment_C", f"{mean_pc1_segments['C']:.4f}")
    ]

    with open("outputs/part3_results.txt", "w") as f:
        for key, val in results:
            f.write(f"{key} {val}\n")

    print("Part 3 completed successfully.")

if __name__ == "__main__":
    run_part3()