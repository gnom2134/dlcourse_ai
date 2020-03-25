import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    f_n = 0
    t_n = 0
    f_p = 0
    t_p = 0

    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i]:
            if prediction[i] == 1:
                t_p += 1
                f_n += 1
            else:
                t_n += 1
        else:
            if prediction[i] == 1:
                t_n += 1
                f_p += 1
            else:
                f_n += 1
    
    precision = (1.) * t_p / (t_p + f_p)
    recall = (1.) * t_p / f_n
    accuracy = (1.) * (t_p + (t_n - f_p)) / (f_n + t_n)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    hit = np.sum(prediction == ground_truth)
    return (1.) * hit / len(ground_truth)
