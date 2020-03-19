def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    count, num_samples = 0, prediction.shape[0]
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            count += 1
    accuracy = count/num_samples
    return accuracy
