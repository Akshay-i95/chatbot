"""
Pre-flight Validation Script - Validates environment and dependencies before running the pipeline

This script checks:
- Python version compatibility
- Required packages and versions
- Azure credentials configuration
- Memory and disk space
- Network connectivity
"""

import sys
import os
import subprocess
import importlib
from typing import List, Tuple, Dict
from dotenv import load_dotenv

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.critical_failures = []

def print_header():
    print("=" * 60)
    print("🔍 PDF VECTORIZATION PIPELINE - PRE-FLIGHT CHECK")
    print("=" * 60)

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print("\n🐍 Checking Python version...")
    
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major >= required_major and version.minor >= required_minor:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python {required_major}.{required_minor}+")
        return False

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed"""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        ('azure-storage-blob', '12.0.0'),
        ('python-dotenv', '0.19.0'),
        ('PyPDF2', '3.0.0'),
        ('pdfplumber', '0.7.0'),
        ('sentence-transformers', '2.0.0'),
        ('chromadb', '0.4.0'),
        ('numpy', '1.20.0'),
        ('pandas', '1.3.0'),
        ('tiktoken', '0.3.0'),
        ('tqdm', '4.60.0')
    ]
    
    missing_packages = []
    all_good = True
    
    for package_name, min_version in required_packages:
        try:
            # Try to import the package
            if package_name == 'azure-storage-blob':
                import azure.storage.blob
                package = azure.storage.blob
            elif package_name == 'python-dotenv':
                import dotenv
                package = dotenv
            elif package_name == 'PyPDF2':
                import PyPDF2
                package = PyPDF2
            elif package_name == 'pdfplumber':
                import pdfplumber
                package = pdfplumber
            elif package_name == 'sentence-transformers':
                import sentence_transformers
                package = sentence_transformers
            elif package_name == 'chromadb':
                import chromadb
                package = chromadb
            elif package_name == 'numpy':
                import numpy
                package = numpy
            elif package_name == 'pandas':
                import pandas
                package = pandas
            elif package_name == 'tiktoken':
                import tiktoken
                package = tiktoken
            elif package_name == 'tqdm':
                import tqdm
                package = tqdm
            else:
                package = importlib.import_module(package_name)
            
            # Get version if available
            version = getattr(package, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
            
        except ImportError:
            print(f"❌ {package_name}: Not installed")
            missing_packages.append(package_name)
            all_good = False
        except Exception as e:
            print(f"⚠️ {package_name}: Error checking - {str(e)}")
            all_good = False
    
    return all_good, missing_packages

def check_environment_config() -> Tuple[bool, List[str]]:
    """Check environment configuration"""
    print("\n⚙️ Checking environment configuration...")
    
    load_dotenv()
    
    required_vars = ['AZURE_STORAGE_ACCOUNT_NAME', 'AZURE_STORAGE_CONTAINER_NAME']
    auth_vars = ['AZURE_STORAGE_CONNECTION_STRING', 'AZURE_STORAGE_SAS_TOKEN', 'AZURE_STORAGE_ACCOUNT_KEY']
    
    missing_vars = []
    issues = []
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Set")
    
    # Check authentication variables (at least one should be set)
    auth_set = any(os.getenv(var) for var in auth_vars)
    if not auth_set:
        print("❌ Authentication: No Azure credentials configured")
        issues.append("No authentication method configured")
    else:
        for var in auth_vars:
            value = os.getenv(var)
            if value:
                print(f"✅ {var}: Set")
            else:
                print(f"⚪ {var}: Not set (optional)")
    
    # Check optional configuration
    optional_vars = {
        'AZURE_BLOB_FOLDER_PATH': 'edipedia/2025-2026/',
        'MAX_WORKERS': '8',
        'CHUNK_SIZE': '1000',
        'VECTOR_DB_TYPE': 'chromadb'
    }
    
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"✅ {var}: {value} {'(default)' if not os.getenv(var) else ''}")
    
    return len(missing_vars) == 0 and len(issues) == 0, missing_vars + issues

def check_system_resources() -> bool:
    """Check system resources"""
    print("\n💻 Checking system resources...")
    
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb >= 4:
            print(f"✅ Available Memory: {available_gb:.1f} GB")
            memory_ok = True
        elif available_gb >= 2:
            print(f"⚠️ Available Memory: {available_gb:.1f} GB (minimum)")
            memory_ok = True
        else:
            print(f"❌ Available Memory: {available_gb:.1f} GB (insufficient)")
            memory_ok = False
        
        # Check disk space
        disk = psutil.disk_usage('.')
        available_disk_gb = disk.free / (1024**3)
        
        if available_disk_gb >= 2:
            print(f"✅ Available Disk: {available_disk_gb:.1f} GB")
            disk_ok = True
        else:
            print(f"❌ Available Disk: {available_disk_gb:.1f} GB (need at least 2GB)")
            disk_ok = False
        
        return memory_ok and disk_ok
        
    except ImportError:
        print("⚠️ psutil not installed, skipping resource check")
        return True
    except Exception as e:
        print(f"⚠️ Could not check system resources: {str(e)}")
        return True

def check_network_connectivity() -> bool:
    """Check network connectivity to required services"""
    print("\n🌐 Checking network connectivity...")
    
    test_urls = [
        "https://edifystorageaccount.blob.core.windows.net",
        "https://huggingface.co"  # For downloading models
    ]
    
    all_good = True
    
    for url in test_urls:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=10)
            print(f"✅ {url}: Accessible")
        except Exception as e:
            print(f"❌ {url}: Not accessible - {str(e)}")
            all_good = False
    
    return all_good

def check_azure_connection() -> bool:
    """Test Azure Blob Storage connection"""
    print("\n☁️ Testing Azure Blob Storage connection...")
    
    try:
        from azure_blob_test import load_config, create_blob_service_client, test_connection
        
        config = load_config()
        if not config.get('account_name') or not config.get('container_name'):
            print("❌ Azure configuration incomplete")
            return False
        
        blob_service_client = create_blob_service_client(config)
        container_client = test_connection(blob_service_client, config['container_name'])
        
        if container_client:
            print("✅ Azure Blob Storage: Connection successful")
            return True
        else:
            print("❌ Azure Blob Storage: Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Azure Blob Storage: Error - {str(e)}")
        return False

def main():
    """Run all validation checks"""
    print_header()
    
    result = ValidationResult()
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("System Resources", check_system_resources),
        ("Network Connectivity", check_network_connectivity),
    ]
    
    # Package check
    print("\n📦 Checking required packages...")
    packages_ok, missing_packages = check_required_packages()
    if packages_ok:
        result.passed += 1
        print("✅ All required packages are installed")
    else:
        result.failed += 1
        result.critical_failures.append("Missing packages")
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
    
    # Environment check
    env_ok, missing_env = check_environment_config()
    if env_ok:
        result.passed += 1
        print("✅ Environment configuration is complete")
    else:
        result.failed += 1
        result.critical_failures.append("Environment configuration")
        print(f"❌ Environment issues: {', '.join(missing_env)}")
    
    # Run other checks
    for check_name, check_func in checks:
        try:
            if check_func():
                result.passed += 1
            else:
                result.failed += 1
                if check_name == "Python Version":
                    result.critical_failures.append(check_name)
        except Exception as e:
            print(f"⚠️ {check_name} check failed: {str(e)}")
            result.warnings += 1
    
    # Azure connection check (only if environment is configured)
    if env_ok:
        if check_azure_connection():
            result.passed += 1
        else:
            result.failed += 1
            result.critical_failures.append("Azure connection")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {result.passed}")
    print(f"❌ Failed: {result.failed}")
    print(f"⚠️ Warnings: {result.warnings}")
    
    if result.critical_failures:
        print(f"\n🚨 Critical Issues:")
        for issue in result.critical_failures:
            print(f"  - {issue}")
        print("\n❌ Please fix critical issues before running the pipeline")
        return False
    elif result.failed > 0:
        print(f"\n⚠️ Some checks failed, but pipeline may still work")
        print("💡 Review the issues above and proceed with caution")
        return True
    else:
        print(f"\n🎉 All checks passed! Ready to run the pipeline")
        print("💡 Run: python main_pipeline.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
