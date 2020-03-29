def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TP, TN, FP, FN = 0, 0, 0, 0
    length = ground_truth.shape[0]
    for i in range(length):

        if prediction[i] == 0:
            if ground_truth[i] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if ground_truth[i] == 1:
                TP += 1
            else:
                FP += 1
    accuracy = (TP + TN) / length
    if TP + FP != 0:
        precision = TP / (TP + FP)
    if TP + FN != 0:
        recall = TP / (TP + FN)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    count, num_samples = 0, prediction.shape[0]
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            count += 1
    accuracy = count/num_samples
    return accuracy
