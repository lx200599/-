import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
os.makedirs('report/figures', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False             # 允许负号正常显示
def train_model(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        train_loss = loss_sum / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # 验证
        model.eval()
        val_total, val_correct, val_loss_sum = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    plot_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)

def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc)+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_acc, 'bo-', label='Train Acc')
    plt.plot(epochs, val_acc, 'go-', label='Val Acc')
    plt.plot(epochs, train_loss, 'ro-', label='Train Loss')
    plt.plot(epochs, val_loss, 'yo-', label='Val Loss')
    plt.title('训练与验证的准确率和损失')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    os.makedirs('report/figures', exist_ok=True)
    plt.savefig('report/figures/accuracy_loss_chart000.png')
    plt.close()

def test_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    os.makedirs('report/figures', exist_ok=True)
    plt.savefig('report/figures/confusion_matrix000.png')
    plt.close()
