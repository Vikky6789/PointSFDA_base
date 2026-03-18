import os
import shutil
import glob

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["FORCE_CUDA"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(BASE_DIR, 'h100_core')
os.makedirs(CORE_DIR, exist_ok=True)

MODULES = [
    "pointnet2_ops_lib",
    "Chamfer3D",
    "extensions/chamfer_dist",
    "extensions/expansion_penalty"
]

for rel_path in MODULES:
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        continue
    
    os.chdir(abs_path)
    os.system("rm -rf build dist *.egg-info")
    os.system("python setup.py build_ext --inplace")
    
    for so_file in glob.glob("**/*.so", recursive=True):
        dest = os.path.join(CORE_DIR, os.path.basename(so_file))
        shutil.copy(so_file, dest)

print(f"All H100 binaries moved to {CORE_DIR}")