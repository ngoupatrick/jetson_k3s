# 0. Test PyTorch + CUDA
print("--- Test PyTorch + CUDA ---")
print("import torch ...")
import torch
# 1. Test PyTorch + CUDA
print("--- Verification PyTorch ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA disponible : {cuda_available}")
if cuda_available:
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPU detectes : {torch.cuda.device_count()}")
    # Test de calcul simple sur GPU
    x = torch.tensor([1.0, 2.0]).cuda()
    print(f"Test calcul GPU reussi : {x.device}")
else:
    print("ERREUR : CUDA n'est pas detecte par PyTorch.")
