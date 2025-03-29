import os
import torch
from torch.autograd import Variable
# ultralytics のセグメンテーション用モデルを安全に読み込むための設定
from ultralytics.nn.tasks import SegmentationModel
torch.serialization.add_safe_globals([SegmentationModel])

# 変換対象の.ptファイルのパスを実行時に入力
pt_path = input("変換対象の.ptファイルのパスを入力してください: ")

# 出力ファイル名を、入力ファイル名に '.onnx' を付加して生成
output_filename = pt_path + '.onnx'

loaded_obj = torch.load(pt_path)
if isinstance(loaded_obj, dict):
    if 'model' in loaded_obj:
        model = loaded_obj['model']
    elif 'state_dict' in loaded_obj:
        model = SegmentationModel()
        model.load_state_dict(loaded_obj['state_dict'])
    else:
        raise ValueError("Checkpoint format is not recognized.")
else:
    model = loaded_obj

# Half 型と Float 型の不一致を防ぐために明示的に float 型にキャストし、評価モードに設定
model = model.float()
model.eval()

# セグメンテーション用の入力サイズに合わせたダミー入力を用意（例：1,3,640,640）
dummy_input = Variable(torch.randn(1, 3, 640, 640))

# 変換後のONNXファイル名は、入力ファイル名に '.onnx' を付加したものとして出力
torch.onnx.export(model, dummy_input, output_filename, verbose=True, opset_version=20)
print(f"ONNXファイルを {output_filename} として出力しました")