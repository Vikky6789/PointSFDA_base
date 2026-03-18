import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["FORCE_CUDA"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "pointnet2_ops_lib",
    "Chamfer3D",
    "extensions/chamfer_dist",
    "extensions/expansion_penalty"
]

print("🚀 Compiling modules IN-PLACE for H100...")
for rel_path in MODULES:
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        continue
    
    print(f"🏗️ Building {rel_path}...")
    os.chdir(abs_path)
    os.system("rm -rf build dist *.egg-info")
    os.system("python setup.py build_ext --inplace")

print("✅ All modules compiled successfully in their original folders!")