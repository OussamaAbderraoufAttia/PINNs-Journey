"""
Run all notebooks as Python scripts.
"""
import sys
import json
import os
from pathlib import Path
import traceback
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
os.chdir(str(Path(__file__).parent.parent))


def extract_code_from_notebook(notebook_path):
    """Extract code cells from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            lines = code.strip().split('\n')
            filtered = [l for l in lines if not l.strip().startswith('%') and not l.strip().startswith('!')]
            code = '\n'.join(filtered)
            if code.strip():
                code_cells.append(code)
    
    return '\n\n'.join(code_cells)


def run_notebook_as_script(notebook_name):
    """Run a notebook by extracting and executing its code."""
    notebook_path = Path(__file__).parent.parent / 'notebooks' / notebook_name
    
    print(f"\n{'='*60}")
    print(f"Running: {notebook_name}")
    print(f"{'='*60}\n")
    
    start = time.time()
    try:
        code = extract_code_from_notebook(notebook_path)
        
        namespace = {
            '__name__': '__main__',
            '__file__': str(notebook_path),
            'sys': sys,
        }
        
        exec(compile(code, str(notebook_path), 'exec'), namespace)
        
        elapsed = time.time() - start
        print(f"  OK: {notebook_name} ({elapsed:.1f}s)")
        return True
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL: {notebook_name} ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return False


def main():
    notebooks = [
        '00_quickstart.ipynb',
        '01_autograd_fundamentals.ipynb',
        '02_first_pinn_ode.ipynb',
        '03_heat_equation.ipynb',
        '04_burgers_equation.ipynb',
        '05_wave_equation.ipynb',
        '06_poisson_equation.ipynb',
        '07_reaction_diffusion.ipynb',
    ]
    
    results = {}
    total_start = time.time()
    for notebook in notebooks:
        results[notebook] = run_notebook_as_script(notebook)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"SUMMARY (total: {total_elapsed:.1f}s)")
    print(f"{'='*60}\n")
    
    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}\n")
    
    for name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {name}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
