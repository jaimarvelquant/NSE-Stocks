#!/usr/bin/env python3
"""Code verification script for zerodha_auto_login_nse.py"""

import sys
import os
from pathlib import Path

def check_syntax():
    """Check Python syntax"""
    print("1. Checking Python syntax...")
    try:
        with open('zerodha_auto_login_nse.py', 'r') as f:
            code = f.read()
        compile(code, 'zerodha_auto_login_nse.py', 'exec')
        print("   ✅ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_imports():
    """Check if required imports are available"""
    print("\n2. Checking imports...")
    required_imports = [
        'json', 'os', 'traceback', 'datetime', 'requests', 'pyotp', 
        'pandas', 'pyarrow', 'kiteconnect', 'logzero', 'threading',
        'sys', 'time', 're', 'webbrowser', 'http.server', 'urllib.parse',
        'logging', 'pathlib', 'shutil', 'subprocess'
    ]
    
    missing = []
    for module in required_imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n   ⚠️  Missing imports: {', '.join(missing)}")
        return False
    return True

def check_paths():
    """Check critical file paths"""
    print("\n3. Checking critical paths...")
    script_dir = os.path.dirname(os.path.abspath('zerodha_auto_login_nse.py'))
    
    paths = {
        'Transform script': Path(script_dir) / 'marvelquant' / 'scripts' / 'transformation' / 'transformers' / 'transform_nautilus.py',
        'Lot size CSV': Path(script_dir) / 'marvelquant' / 'NseLotSize.csv',
        'Stock names CSV': Path(script_dir) / 'stock_namess.csv',
        'Holidays CSV': Path(script_dir) / 'nse_holidays.csv',
    }
    
    all_ok = True
    for name, path in paths.items():
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} - MISSING")
            all_ok = False
    
    # Check directories (will be created if missing)
    dirs = {
        'Input directory': Path('/home/ubuntu/raw_data'),
        'Output directory': Path('/media/ubuntu/Dataspace/nautilus_modal'),
    }
    
    for name, path in dirs.items():
        if path.exists():
            print(f"   ✅ {name}: {path} (exists)")
        else:
            print(f"   ⚠️  {name}: {path} (will be created)")
    
    # Check Python path
    python_path = Path('/usr/bin/python3')
    if python_path.exists():
        print(f"   ✅ Python path: {python_path}")
    else:
        print(f"   ❌ Python path: {python_path} - MISSING")
        all_ok = False
    
    return all_ok

def check_indentation():
    """Check for common indentation issues"""
    print("\n4. Checking indentation...")
    try:
        with open('zerodha_auto_login_nse.py', 'r') as f:
            lines = f.readlines()
        
        issues = []
        for i, line in enumerate(lines, 1):
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                issues.append(f"Line {i}: Mixed tabs and spaces")
            
            # Check for inconsistent indentation (basic check)
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(stripped)
                if indent % 4 != 0 and indent > 0:
                    # This is just a warning, not necessarily an error
                    pass
        
        if issues:
            print(f"   ⚠️  Found {len(issues)} potential indentation issues")
            for issue in issues[:5]:  # Show first 5
                print(f"      {issue}")
            if len(issues) > 5:
                print(f"      ... and {len(issues) - 5} more")
        else:
            print("   ✅ No obvious indentation issues found")
        
        return len(issues) == 0
    except Exception as e:
        print(f"   ❌ Error checking indentation: {e}")
        return False

def check_nautilus_function():
    """Check Nautilus transformation function structure"""
    print("\n5. Checking Nautilus transformation function...")
    try:
        with open('zerodha_auto_login_nse.py', 'r') as f:
            content = f.read()
        
        checks = {
            'Function definition': 'def run_nautilus_transformation' in content,
            'Error handling': 'except subprocess.TimeoutExpired' in content,
            'Exception handling': 'except Exception as e' in content,
            'Telegram alerts': 'send_telegram_alert' in content,
            'Timeout setting': 'timeout=21600' in content,
            'Path construction': 'Path(script_dir)' in content,
            'Subprocess call': 'subprocess.run' in content,
        }
        
        all_ok = True
        for check, result in checks.items():
            if result:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Code Verification for zerodha_auto_login_nse.py")
    print("=" * 60)
    
    results = []
    results.append(check_syntax())
    results.append(check_imports())
    results.append(check_paths())
    results.append(check_indentation())
    results.append(check_nautilus_function())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL CHECKS PASSED - Code appears ready for execution")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues above")
        return 1

if __name__ == '__main__':
    sys.exit(main())

