import os
import subprocess
import glob
import sys

def install_wheels(wheel_dir):
    """Install all wheel files in the specified directory"""
    print(f"切換到目錄: {wheel_dir}")
    os.chdir(wheel_dir)
    
    # 找到所有.whl文件
    wheel_files = glob.glob("*.whl")
    total_wheels = len(wheel_files)
    
    if total_wheels == 0:
        print("找不到任何.whl文件")
        return
    
    print(f"找到 {total_wheels} 個wheel文件")
    successful = []
    failed = []
    
    # 安裝每個wheel文件
    for i, wheel in enumerate(wheel_files, 1):
        print(f"\n[{i}/{total_wheels}] 正在安裝: {wheel}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", wheel],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                successful.append(wheel)
                print(f"✓ 成功安裝: {wheel}")
            else:
                failed.append((wheel, result.stderr))
                print(f"✗ 安裝失敗: {wheel}")
                print("錯誤信息:", result.stderr)
                
        except Exception as e:
            failed.append((wheel, str(e)))
            print(f"✗ 安裝失敗: {wheel}")
            print("錯誤信息:", str(e))
    
    # 顯示安裝結果
    print(f"\n=== 安裝完成 ===")
    print(f"成功: {len(successful)}/{total_wheels}")
    print(f"失敗: {len(failed)}/{total_wheels}")
    
    if failed:
        print("\n失敗的套件:")
        for wheel, error in failed:
            print(f"- {wheel}")
        
        # 保存錯誤日誌
        log_file = "wheel_install_errors.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for wheel, error in failed:
                f.write(f"\n=== {wheel} ===\n")
                f.write(error)
                f.write("\n")
        print(f"\n詳細錯誤日誌已保存到: {log_file}")

if __name__ == "__main__":
    # 自動選擇當前目錄的pipcompiled
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wheel_dir = os.path.join(current_dir, "setup","pipcompiled")
    
    if not wheel_dir:
        print("目錄遺失")
        sys.exit(1)
    
    install_wheels(wheel_dir)