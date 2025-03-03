import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

def is_pytorch_package(package):
    """Check if package is part of PyTorch ecosystem"""
    pytorch_packages = ['torch', 'torchvision', 'torchaudio']
    package_name = package.split('==')[0]
    return any(pkg in package_name for pkg in pytorch_packages)

def create_wheel(package, output_dir):
    try:
        # Check if it's a PyTorch ecosystem package with CUDA
        if is_pytorch_package(package) and '+cu' in package:
            # Extract version and CUDA version
            package_name = package.split('==')[0]
            version_cuda = package.split('==')[1]
            base_version = version_cuda.split('+')[0]
            cuda_version = version_cuda.split('+')[1]
            
            # Use PyTorch index URL
            index_url = f"https://download.pytorch.org/whl/{cuda_version}"
            print(f"Using PyTorch index URL: {index_url}")
            
            cmd = [
                sys.executable, "-m", "pip", "wheel",
                "--no-deps",
                "--wheel-dir", output_dir,
                "-i", index_url,
                f"{package_name}=={base_version}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, result.stdout + "\n" + result.stderr
            return True, None
            
        else:
            # Handle regular packages
            result = subprocess.run([
                sys.executable, "-m", "pip", "wheel",
                "--no-deps",
                "--wheel-dir", output_dir,
                package
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, result.stdout + "\n" + result.stderr
            return True, None
            
    except Exception as e:
        return False, str(e)

# 創建 Tkinter 根窗口
root = tk.Tk()
root.withdraw()  # 隱藏主窗口

# 選擇 requirements.txt 文件
requirements_file = filedialog.askopenfilename(
    title="選擇 requirements.txt 文件",
    filetypes=[("Text files", "*.txt")]
)
if not requirements_file:
    print("未選擇 requirements.txt 文件")
    sys.exit(1)

# 選擇保存 .whl 文件的目錄
output_dir = filedialog.askdirectory(title="選擇保存 .whl 文件的目錄")
if not output_dir:
    print("未選擇保存 .whl 文件的目錄")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

# 讀取並處理套件
with open(requirements_file, "r", encoding="utf-8") as req_file:  # 明確指定 UTF-8 編碼
    packages = [line.strip() for line in req_file if line.strip()]

total_packages = len(packages)
successful_packages = []
failed_packages = []

print(f"開始處理 {total_packages} 個套件...")

for i, package in enumerate(packages, 1):
    print(f"\n[{i}/{total_packages}] 正在處理套件: {package}")
    
    success, error_msg = create_wheel(package, output_dir)
    
    if success:
        successful_packages.append(package)
        print(f"✓ 成功創建 wheel: {package}")
    else:
        failed_packages.append((package, error_msg))
        print(f"✗ 處理失敗: {package}")
        print("錯誤信息:", error_msg)

# 顯示總結
print("\n=== 處理完成 ===")
print(f"成功: {len(successful_packages)}/{total_packages}")
print(f"失敗: {len(failed_packages)}/{total_packages}")

if failed_packages:
    print("\n失敗的套件:")
    for package, error in failed_packages:
        print(f"- {package}")
    
    # 將錯誤日誌保存到文件
    log_file = os.path.join(output_dir, "failed_packages.log")
    with open(log_file, "w", encoding="utf-8") as f:  # 使用 UTF-8 保存錯誤日誌
        for package, error in failed_packages:
            f.write(f"\n=== {package} ===\n")
            f.write(error)
            f.write("\n")
    print(f"\n詳細錯誤日誌已保存到: {log_file}")

# 顯示完成消息
print("處理完成\n成功: {}\n失敗: {}".format(len(successful_packages), len(failed_packages)))
