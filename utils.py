import numpy as np


def accuracy(y_true, y_pred):
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_predictions += 1
    # returns accuracy
    return correct_predictions / len(y_true)


def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:
            fn += 1
    return fn


def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize precision to 0
    precision = 0
    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision
    # calculate and return average precision over all classes
    precision /= num_classes
    return precision


def micro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize tp and fp to 0
    tp = 0
    fp = 0
    # loop over all classes
    for class_ in list(set(y_true)):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)
    # calculate and return overall precision
    precision = tp / (tp + fp + 0.1)
    return precision


def macro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize recall to 0
    recall = 0
    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)
        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)
        # keep adding recall for all classes
        recall += temp_recall
    # calculate and return average recall over all classes
    recall /= num_classes
    return recall


def micro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize tp and fp to 0
    tp = 0
    fn = 0
    # loop over all classes
    for class_ in list(set(y_true)):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        # calculate false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)
    # calculate and return overall recall
    recall = tp / (tp + fn + 0.1)
    return recall


def macro_f1(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize f1 to 0
    f1 = 0
    # loop over all classes
    for class_ in list(set(y_true)):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
        # keep adding f1 score for all classes
        f1 += temp_f1
    # calculate and return average f1 score over all classes
    f1 /= num_classes

    return f1


def micro_f1(y_true, y_pred):
    # micro-averaged precision score
    P = micro_precision(y_true, y_pred)
    # micro-averaged recall score
    R = micro_recall(y_true, y_pred)
    # micro averaged f1 score
    f1 = 2*P*R / (P + R + 0.1)
    return f1

