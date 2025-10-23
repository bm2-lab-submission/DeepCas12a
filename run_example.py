import torch
from DeepCas12a import Episgt, VisionTransformer

data = Episgt('example/example_sequences.txt', num_epi_features=2, with_y=True)
X, y = data.get_dataset()
X = torch.tensor(X).unsqueeze(1)
# 你也可以选择其他fold: fold_2, fold_3, ..., fold_9或进行集成预测
model = VisionTransformer()
model.load_state_dict(torch.load('trained_model/fold1.pth'))
model.eval()
with torch.no_grad():
    predictions = model(X)
