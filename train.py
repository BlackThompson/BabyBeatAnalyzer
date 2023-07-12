# _*_ coding : utf-8 _*_
# @Time : 2023/7/5 21:00
# @Author : Black
# @File : train
# @Project : BabyBeatAnalyzer

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import MyDataset
import utils.config as config
from models.RNN.model import RNN
from data.utils import collate_fn
from evaluate import Acurracy, TPR


def train(model, criterion, optimizer, train_loader, val_loader,
          num_epochs, early_stop_patience, model_path, device):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            # 切记要将labels转换为tensor
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping! Total epochs: {epoch + 1}")
                break


def evaluate(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            predictions.append(predicted.tolist())
            true_labels.append(labels.tolist())

        # test_accuracy = test_correct / test_total
        # print(f"Test Accuracy: {test_accuracy:.4f}")

    test_accuracy = Acurracy(predictions, true_labels)
    test_tpr = TPR(predictions, true_labels)

    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"TPR: {test_tpr:.4f}")


if __name__ == '__main__':
    train_path = r'./data/dataset/train_4519.xlsx'
    val_path = r'./data/dataset/val_503.xlsx'
    test_path = r'./data/dataset/test_558.xlsx'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train on', device)

    # 加载数据
    train_data = pd.read_excel(train_path)
    val_data = pd.read_excel(val_path)
    test_data = pd.read_excel(test_path)
    train_loader = DataLoader(MyDataset(train_data['fhr'], train_data['tag']), batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MyDataset(val_data['fhr'], val_data['tag']), batch_size=1, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(MyDataset(test_data['fhr'], test_data['tag']), batch_size=1, shuffle=False,
                             collate_fn=collate_fn)

    # 创建模型

    model = RNN(config.input_size, config.hidden_size, config.output_size, config.num_layers, type='GRU')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 模型路径
    # 记得修改！！！！！
    model_path = r'./models/RNN/trained_models/GRU.pth'

    # 训练模型
    train(model, criterion, optimizer, train_loader, val_loader,
          config.epochs, config.patience, model_path, device)

    # 测试模型
    model.load_state_dict(torch.load(model_path))
    evaluate(model, test_loader, device)
