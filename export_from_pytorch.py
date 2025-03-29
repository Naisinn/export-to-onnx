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
loaded_obj = torch.load(pt_path)
if isinstance(loaded_obj, dict):
    if 'model' in loaded_obj:
        vgg16 = loaded_obj['model']
    elif 'state_dict' in loaded_obj:
        vgg16 = SegmentationModel()
        vgg16.load_state_dict(loaded_obj['state_dict'])
    else:
        raise ValueError("Checkpoint format is not recognized.")
else:
    vgg16 = loaded_obj

vgg16.eval()

x = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(vgg16, x, 'vgg16_pytorch.onnx', verbose=True, opset_version=10)