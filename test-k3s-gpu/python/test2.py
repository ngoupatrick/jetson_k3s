# 0. Test PyTorch + CUDA
print("--- Test PyTorch + CUDA ---")
print("import torch ...")

import torch
import time

# 1. Verification de la disponibilite de CUDA
if not torch.cuda.is_available():
    print("Erreur : CUDA n'est pas detecte. Verifiez votre installation PyTorch pour Jetson.")
    exit()

device = torch.device("cuda")
print(f"Appareil utilise : {torch.cuda.get_device_name(0)}")

# 2. Creation de deux grandes matrices (ex: 3000x3000)
size = 3000
cpu_a = torch.randn(size, size)
cpu_b = torch.randn(size, size)

# --- TEST SUR CPU ---
start_cpu = time.time()
cpu_res = torch.matmul(cpu_a, cpu_b)
end_cpu = time.time()
print(f"Temps CPU : {end_cpu - start_cpu:.4f} secondes")

# --- TEST SUR GPU (Jetson CUDA) ---
# Transfert des donnees vers le GPU
gpu_a = cpu_a.to(device)
gpu_b = cpu_b.to(device)

# "Prechauffage" du GPU (important pour les mesures de temps)
_ = torch.matmul(gpu_a, gpu_b)
torch.cuda.synchronize()

start_gpu = time.time()
gpu_res = torch.matmul(gpu_a, gpu_b)
# On attend que le GPU finisse le calcul avant de stopper le chrono
torch.cuda.synchronize() 
end_gpu = time.time()

print(f"Temps GPU : {end_gpu - start_gpu:.4f} secondes")

# 3. Calcul de l'acceleration
speedup = (end_cpu - start_cpu) / (end_gpu - start_gpu)
print(f"\nLe GPU est {speedup:.1f}x plus rapide que le CPU sur ce calcul !")
