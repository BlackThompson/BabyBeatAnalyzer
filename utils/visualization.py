# _*_ coding : utf-8 _*_
# @Time : 2023/7/7 9:09
# @Author : Black
# @File : visualization
# @Project : BabyBeatAnalyzer

import matplotlib.pyplot as plt


def show_data(data, y_start=0, y_end=200, title_name='title', x_label='Time', y_label='Y'):
    """
    Show data in a line chart
    :param data:
    :param y_start:
    :param y_end:
    :param title_name:
    :param x_label:
    :param y_label:
    :return:
    """

    # 创建时间轴
    time = range(len(data))

    # 绘制折线图
    plt.plot(time, data)

    # 设置纵坐标范围
    plt.ylim(y_start, y_end)

    # 添加标题和轴标签
    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 显示图形
    plt.show()
