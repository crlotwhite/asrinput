#!/usr/bin/env python3
"""
Build script for ASRInput - Creates standalone executable using PyInstaller
"""

import subprocess
import sys
import shutil
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True if sys.platform == "win32" else False)
    return result.returncode == 0


def get_customtkinter_path() -> str:
    """Get the path to customtkinter package."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", "import customtkinter; print(customtkinter.__path__[0])"],
        capture_output=True,
        text=True,
        shell=True if sys.platform == "win32" else False
    )
    return result.stdout.strip()


def build():
    """Build the executable."""
    app_name = "asrinput"
    main_file = "main.py"
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("  ASRInput Build Script")
    print("="*60)
    
    # Step 1: Sync dependencies
    if not run_command(["uv", "sync"], "Syncing dependencies..."):
        print("❌ Failed to sync dependencies")
        return False
    
    # Step 2: Install pyinstaller
    if not run_command(["uv", "add", "--dev", "pyinstaller"], "Installing PyInstaller..."):
        print("❌ Failed to install PyInstaller")
        return False
    
    # Step 3: Get customtkinter path for data inclusion
    print("\n📦 Getting customtkinter path...")
    ctk_path = get_customtkinter_path()
    if not ctk_path:
        print("❌ Failed to get customtkinter path")
        return False
    print(f"   Found: {ctk_path}")
    
    # Step 4: Build with PyInstaller
    pyinstaller_cmd = [
        "uv", "run", "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", app_name,
        f"--add-data={ctk_path};customtkinter",
        main_file
    ]
    
    if not run_command(pyinstaller_cmd, "Building executable with PyInstaller..."):
        print("❌ Build failed")
        return False
    
    # Step 5: Check output
    exe_path = Path("dist") / f"{app_name}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print("\n" + "="*60)
        print("  ✅ Build Successful!")
        print("="*60)
        print(f"   Output: {exe_path.absolute()}")
        print(f"   Size: {size_mb:.1f} MB")
        print("\n⚠️  Note: First run may take time to download Whisper model (~500MB)")
        return True
    else:
        print("❌ Executable not found after build")
        return False


def clean():
    """Clean build artifacts."""
    dirs_to_remove = ["build", "dist", "__pycache__"]
    files_to_remove = list(Path(".").glob("*.spec"))
    
    print("\n🧹 Cleaning build artifacts...")
    
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_name}/")
    
    for file_path in files_to_remove:
        file_path.unlink()
        print(f"   Removed: {file_path}")
    
    print("✅ Clean complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ASRInput executable")
    parser.add_argument("command", nargs="?", default="build", 
                        choices=["build", "clean", "help"],
                        help="Command to run (default: build)")
    
    args = parser.parse_args()
    
    if args.command == "build":
        success = build()
        sys.exit(0 if success else 1)
    elif args.command == "clean":
        clean()
    elif args.command == "help":
        parser.print_help()


if __name__ == "__main__":
    main()
