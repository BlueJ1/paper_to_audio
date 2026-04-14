#!/usr/bin/env python3
"""Verify that all dependencies and configuration are correct."""

import sys
import os

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package} is installed")
        return True
    except ImportError:
        print(f"✗ {package} is NOT installed - run: pip install {package}")
        return False

def check_command(command):
    """Check if a command is available on the system."""
    import shutil
    if shutil.which(command):
        print(f"✓ {command} is installed")
        return True
    else:
        print(f"✗ {command} is NOT installed - see SETUP.md for installation instructions")
        return False

def check_env_var(var_name):
    """Check if an environment variable is set."""
    from dotenv import load_dotenv
    load_dotenv()
    value = os.getenv(var_name, "").strip()
    if value:
        print(f"✓ {var_name} is set")
        return True
    else:
        print(f"✗ {var_name} is NOT set - add to .env file")
        return False

def main():
    print("=== Checking Dependencies ===\n")

    all_ok = True

    print("Python Packages:")
    all_ok &= check_import("langchain")
    all_ok &= check_import("langchain_google_genai", "langchain-google-genai")
    all_ok &= check_import("langchain_community", "langchain-community")
    all_ok &= check_import("pypdf")
    all_ok &= check_import("requests")
    all_ok &= check_import("dotenv", "python-dotenv")
    all_ok &= check_import("pydub")

    print("\nSystem Commands:")
    all_ok &= check_command("ffmpeg")

    print("\nEnvironment Variables:")
    all_ok &= check_env_var("GOOGLE_API_KEY")
    all_ok &= check_env_var("MURF_API_KEY")

    print("\n" + "="*40)
    if all_ok:
        print("✓ All checks passed! You're ready to go.")
        print("\nRun: python main.py papers/Titans.pdf --out output.mp3")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nSee SETUP.md for detailed installation instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
