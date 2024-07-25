
import torch


# GPU 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)