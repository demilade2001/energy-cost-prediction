# -*- coding: utf-8 -*-
"""
MASTER RUNNER - Orchestrates all data analytics scripts
Executes: Preparation → EDA → ML Model Training

@author: Demilade
@created: 2026-05-20

Usage:
    python MASTER_RUNNER.py              # Run all steps
    python MASTER_RUNNER.py --no-eda    # Skip EDA visualizations
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPTS_DIR = Path(__file__).parent
SCRIPT_EXECUTION_ORDER = [
    ("Preparation Solution.py", "Data Preparation & Merging"),
    ("EDA.py", "Exploratory Data Analysis"),
    ("ML MODEL.py", "ML Model Training & Comparison"),
]

def print_header(title):
    """Print a formatted section header."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")

def print_step(step_num, total, script_name, description):
    """Print step header."""
    print(f"\n[{step_num}/{total}] Running: {description}")
    print(f"      Script: {script_name}")
    print("-" * 80)

def run_script(script_path):
    """
    Execute a Python script and capture output.
    Returns: (success: bool, runtime: float, error: str)
    """
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600  # 10 minute timeout
        )
        runtime = time.time() - start
        
        if result.returncode == 0:
            return True, runtime, None
        else:
            return False, runtime, f"Script exited with code {result.returncode}"
    except subprocess.TimeoutExpired:
        runtime = time.time() - start
        return False, runtime, "Script execution timed out (>600s)"
    except Exception as e:
        runtime = time.time() - start
        return False, runtime, str(e)

def main():
    """Main orchestration function."""
    skip_eda = "--no-eda" in sys.argv
    
    print_header("MASTER DATA ANALYTICS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {SCRIPTS_DIR}")
    if skip_eda:
        print("⚠️  EDA visualizations will be SKIPPED")
    
    results = []
    total_start = time.time()
    
    for idx, (script_name, description) in enumerate(SCRIPT_EXECUTION_ORDER, 1):
        # Skip EDA if requested
        if skip_eda and "EDA" in script_name:
            print(f"\n[{idx}/3] SKIPPED: {description} (--no-eda flag)")
            results.append({
                "script": script_name,
                "description": description,
                "status": "SKIPPED",
                "runtime": 0
            })
            continue
        
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            print(f"❌ ERROR: {script_name} not found in {SCRIPTS_DIR}")
            results.append({
                "script": script_name,
                "description": description,
                "status": "NOT FOUND",
                "runtime": 0
            })
            continue
        
        print_step(idx, 3, script_name, description)
        
        success, runtime, error = run_script(script_path)
        
        if success:
            print(f"✓ COMPLETED in {runtime:.2f}s")
            results.append({
                "script": script_name,
                "description": description,
                "status": "✓ SUCCESS",
                "runtime": runtime
            })
        else:
            print(f"❌ FAILED after {runtime:.2f}s")
            print(f"   Error: {error}")
            results.append({
                "script": script_name,
                "description": description,
                "status": "❌ FAILED",
                "runtime": runtime,
                "error": error
            })
    
    # Print summary
    total_runtime = time.time() - total_start
    print_header("EXECUTION SUMMARY")
    
    for r in results:
        status = r["status"]
        runtime_str = f"{r['runtime']:.2f}s" if r["runtime"] > 0 else "—"
        print(f"{status:<15} | {r['description']:<40} | {runtime_str}")
        if "error" in r:
            print(f"{'':15} | Error: {r['error']}")
    
    print(f"\nTotal execution time: {total_runtime:.2f}s ({total_runtime/60:.2f} minutes)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if all passed
    all_success = all(r["status"] == "✓ SUCCESS" or r["status"] == "SKIPPED" for r in results)
    
    if all_success:
        print("\n🎉 All steps completed successfully!")
        return 0
    else:
        print("\n⚠️  Some steps failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
