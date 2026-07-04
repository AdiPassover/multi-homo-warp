import cv2
import numpy as np

def calc_transfer_error(H, pts1, pts2):
    n = pts1.shape[1]
    pts1_h = np.vstack((pts1, np.ones((1, n))))
    pts3 = H @ pts1_h
    pts3 = pts3[:2, :] / (pts3[2, :] + 1e-8)
    d1 = np.sqrt(np.sum((pts2 - pts3)**2, axis=0))

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)
        
    pts2_h = np.vstack((pts2, np.ones((1, n))))
    pts4 = H_inv @ pts2_h
    pts4 = pts4[:2, :] / (pts4[2, :] + 1e-8)
    d2 = np.sqrt(np.sum((pts1 - pts4)**2, axis=0))

    return (d1 + d2) / 2.0

def multi_homo_fitting(matches_1, matches_2, img1, img2, labels1, init_H, parameters):
    """
    PEaRL-based multi-homography fitting using gco-python.
    """
    para_dist = parameters.get('dist', 5)
    para_lambda = parameters.get('lambda', 20)
    para_beta = parameters.get('beta', 10)
    para_maxdatacost = parameters.get('maxdatacost', 1e4)
    para_gamma = parameters.get('gamma', 2e2)
    
    pts1 = matches_1
    pts2 = matches_2
    num_sites = pts1.shape[1]
    
    from scipy.spatial import Delaunay
    neighbors = np.zeros((num_sites, num_sites), dtype=np.int32)
    
    if num_sites >= 4:
        tri = Delaunay(pts1.T)
        for t in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    p1 = t[i]
                    p2 = t[j]
                    
                    y1, x1 = int(round(pts1[1, p1])), int(round(pts1[0, p1]))
                    y2, x2 = int(round(pts1[1, p2])), int(round(pts1[0, p2]))
                    
                    # Ensure indices are within bounds
                    y1, x1 = np.clip(y1, 0, labels1.shape[0]-1), np.clip(x1, 0, labels1.shape[1]-1)
                    y2, x2 = np.clip(y2, 0, labels1.shape[0]-1), np.clip(x2, 0, labels1.shape[1]-1)
                    
                    l1 = labels1[y1, x1]
                    l2 = labels1[y2, x2]
                    
                    if l1 == l2:
                        neighbors[p1, p2] = 1
                        neighbors[p2, p1] = 1
        
    num_labels = len(init_H)
    
    # data_cost: (num_sites, num_labels + 1)
    data_cost = np.zeros((num_sites, num_labels + 1), dtype=np.int32)
    for i, H in enumerate(init_H):
        dist_err = calc_transfer_error(H, pts1, pts2)
        data_cost[:, i] = np.minimum(dist_err, para_maxdatacost).astype(np.int32)
    data_cost[:, num_labels] = int(para_gamma) # outlier model
    
    smooth_cost = para_lambda * (np.ones((num_labels + 1, num_labels + 1)) - np.eye(num_labels + 1))
    smooth_cost = smooth_cost.astype(np.int32)
    
    try:
        import gco
        edges = np.column_stack(np.where(np.triu(neighbors) > 0)).astype(np.int32)
        # Using gco.cut_general_graph or similar if available
        # Note: gco-python api expects edge weights.
        edge_weights = np.ones(len(edges), dtype=np.int32)
        
        # If gco has a general graph cut:
        # Since gco APIs differ slightly depending on the package version (e.g. gco-python vs pygco)
        # we try standard usage:
        labels = gco.cut_general_graph(edges, edge_weights, data_cost, smooth_cost)
        
    except ImportError:
        print("Warning: gco-python not installed. Using naive argmin for multi-homo-fitting.")
        labels = np.argmin(data_cost, axis=1)
    except AttributeError:
        # If cut_general_graph is not the right function
        print("Warning: gco method cut_general_graph not found. Using naive argmin.")
        labels = np.argmin(data_cost, axis=1)
        
    # Extract Homographies
    multi_homos = []
    cell_matches = []
    
    for i in range(num_labels):
        idx = np.where(labels == i)[0]
        if len(idx) >= 4:
            inliers1 = pts1[:, idx]
            inliers2 = pts2[:, idx]
            H_ref, _ = cv2.findHomography(inliers1.T, inliers2.T, 0)
            if H_ref is not None:
                dist_i = calc_transfer_error(H_ref, inliers1, inliers2)
                inner_ind = dist_i <= para_dist
                if np.sum(inner_ind) >= 4:
                    H_ref_filtered, _ = cv2.findHomography(inliers1[:, inner_ind].T, inliers2[:, inner_ind].T, 0)
                    if H_ref_filtered is not None:
                        multi_homos.append(H_ref_filtered.flatten())
                        cell_matches.append((inliers1[:, inner_ind].T, inliers2[:, inner_ind].T))
    
    return np.array(multi_homos), cell_matches
