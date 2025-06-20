# MNIST 手写数字识别项目

本项目使用卷积神经网络（CNN）在 MNIST 数据集上实现手写数字图像的自动识别，配合《人工智能》课程研究报告使用。

## 项目结构
- `main.py`：主入口，训练 + 测试模型
- `model/cnn_model.py`：CNN 模型结构
- `model/train_utils.py`：训练、测试及图表绘制工具
- `report/figures/`：训练过程图表自动保存在此

## 快速开始
```bash
pip install -r requirements.txt
python main.py
```

## 数据集
- MNIST 数据集通过 `torchvision.datasets.MNIST` 自动下载。

## 输出图表
- 准确率与损失曲线图：`accuracy_loss_chart000.png`
- 混淆矩阵图：`confusion_matrix000.png`
