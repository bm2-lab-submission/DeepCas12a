import torch
from DeepCas12a import Episgt, VisionTransformer

data = Episgt('example/example_sequences.txt', num_epi_features=2, with_y=True)
X, y = data.get_dataset()
X = torch.tensor(X).unsqueeze(1)
# You can also choose other folds: fold_2, fold_3, ..., fold_9 or make ensemble prediction
model = VisionTransformer()
model.load_state_dict(torch.load('trained_model/fold1.pth'))
model.eval()
with torch.no_grad():
    predictions = model(X)
