import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

# 安全なグローバルとしてultralyticsのSegmentationModelを追加
from ultralytics.nn.tasks import SegmentationModel
torch.serialization.add_safe_globals([SegmentationModel])

# 変換対象の.ptファイルのパスを実行後に入力
pt_path = input("変換対象の.ptファイルのパスを入力してください: ")
vgg16 = torch.load(pt_path)
vgg16.eval()

x = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(vgg16, x, 'vgg16_pytorch.onnx', verbose=True, opset_version=10)