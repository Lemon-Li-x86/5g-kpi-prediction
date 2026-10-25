"""Read-only project inventory; preserve hashes before/after algorithm changes."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final_audit_20261025'


def digest(paths):
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / 'app.py', *ROOT.glob('src/ui/**/*.py'),
                 *ROOT.glob('src/services/**/*.py'), *ROOT.glob('src/core/**/*.py')]
    history = [p for folder in ('reports', 'figures') for p in (ROOT / folder).rglob('*')
               if 'final_audit_20261025' not in p.parts and p.is_file()]
    manifest = {'protected_web_integration': digest(protected), 'historical_results': digest(history)}
    baseline = OUT / 'baseline_manifest.json'
    if not baseline.exists():
        baseline.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    else:
        old = json.loads(baseline.read_text(encoding='utf-8'))
        # Compare only original paths: new experiment documents may be added.
        changed = {kind: [p for p, h in old[kind].items() if manifest[kind].get(p) != h]
                   for kind in old}
        (OUT / 'preservation_check.json').write_text(json.dumps(changed, indent=2), encoding='utf-8')
        if any(changed.values()):
            raise RuntimeError(f'Protected/historical files changed: {changed}')
    sources = list(ROOT.glob('src/**/*.py')) + [ROOT / 'app.py']
    texts = {str(p.relative_to(ROOT)).replace('\\', '/'): p.read_text(encoding='utf-8') for p in sources}
    rows = []
    for path, source in texts.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            name, value = '', None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, ast.get_source_segment(source, node.value)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    name = str(node.args[0].value)
                    value = next((ast.get_source_segment(source, k.value) for k in node.keywords if k.arg == 'default'), '')
            if name and value is not None and len(str(value)) < 500:
                refs = [f'{p}:{i}' for p, s in texts.items() for i, line in enumerate(s.splitlines(), 1)
                        if name.lstrip('-').replace('-', '_') in line]
                rows.append({'parameter': name, 'value_or_expression': value, 'definition': f'{path}:{node.lineno}',
                             'references': ' | '.join(refs[:80]),
                             'suggested_owner': path.split('/')[1] if path.startswith('src/') else 'ui'})
    with (OUT / 'parameter_inventory.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    status = subprocess.run(['git', 'status', '-sb'], cwd=ROOT, capture_output=True, text=True).stdout
    (OUT / 'git_status_latest.txt').write_text(status, encoding='utf-8')
    print(f'Inventory: {len(rows)} definitions; protected {len(manifest["protected_web_integration"])} files')


if __name__ == '__main__':
    main()
