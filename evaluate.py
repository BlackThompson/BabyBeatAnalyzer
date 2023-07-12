# _*_ coding : utf-8 _*_
# @Time : 2023/7/5 21:00
# @Author : Black
# @File : evaluate
# @Project : BabyBeatAnalyzer

def Acurracy(predict, true):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    :param predict:
    :param true:
    :return:
    """
    i = 0
    for j in range(len(predict)):
        if predict[j] == true[j]:
            i += 1
    return i / len(predict)


def TPR(predict, true):
    """
    TPR = TP / (TP + FN)
    :param predict:
    :param true:
    :return:
    """
    TP = 0
    FN = 0
    for i in range(len(predict)):
        if predict[i] == 1 and true[i] == 1:
            TP += 1
        elif predict[i] == 0 and true[i] == 1:
            FN += 1
    return TP / (TP + FN)

