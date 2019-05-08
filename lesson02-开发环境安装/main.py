import  torch

print(torch.__version__)
print('gpu:', torch.cuda.is_available())