def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    """
    Predicts labels for test images using nearest neighbor classification.

    Input:
        train_image_feats: N x d matrix, where d is the feature dimensionality
        train_labels: List of N strings, the ground truth labels for training images
        test_image_feats: M x d matrix, where d is the feature dimensionality

    Output:
        test_predicts: List of M strings, predicted labels for test images
    """
    test_predicts = []

    for test_feat in test_image_feats:
        min_dist = float('inf')
        label_final = None  # Initialize to a default value

        for train_feat, train_label in zip(train_image_feats, train_labels):
            dist = np.linalg.norm(test_feat - train_feat)  # Compute L2 distance
            if dist < min_dist:
                min_dist = dist
                label_final = train_label  # Assign the label of the nearest neighbor

        # Ensure a label is assigned
        if label_final is not None:
            test_predicts.append(label_final)
        else:
            raise ValueError("No label assigned for a test feature. Check your data.")

    return test_predicts
