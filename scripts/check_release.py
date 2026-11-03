"""Check the exact staged file list without printing any matched secret values."""
from pathlib import Path
import argparse
import ast
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TEXT={'.py','.json','.yaml','.yml','.md','.txt','.sh','.toml','.html','.js','.css'}
BLOCKED={'.pt','.pth','.safetensors','.hdf5','.h5','.mp4','.npz','.npy','.dpapi','.pem','.key','.zip'}
PATTERNS={
    'private_key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'github_token':re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    'literal_credential':re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*["'](?:sk-[A-Za-z0-9_-]{20,}|[A-Za-z0-9_-]{32,})["']'''),
    'private_workspace':re.compile(r'(?i)[CD]:[/\\](?:code[/\\]|Users[/\\]|具身数据)'),
    'personal_tailnet':re.compile(r'100\.126\.148\.122'),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
    paths=[p for p in result.stdout.decode('utf-8').split('\0') if p]
    problems=[];total=0;maximum=0;python_files=0
    for name in paths:
        path=ROOT/name
        if not path.is_file():problems.append({'path':name,'kind':'missing_file'});continue
        size=path.stat().st_size;total+=size;maximum=max(maximum,size)
        if size>=100*1024**2:problems.append({'path':name,'kind':'github_file_size_limit'})
        if path.suffix.lower() in BLOCKED or path.name.startswith('.env') and path.name!='.env.example':
            problems.append({'path':name,'kind':'excluded_data_weight_or_credential_file'})
        if path.suffix.lower() not in TEXT:continue
        content=path.read_text(encoding='utf-8-sig')
        if path.suffix=='.py':
            python_files+=1
            try:ast.parse(content,filename=name)
            except SyntaxError as error:problems.append({'path':name,'kind':'python_syntax','line':error.lineno})
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                problems.append({'path':name,'kind':kind,'line':content[:match.start()].count('\n')+1})
    report={'scope':'exact_git_index_paths_current_file_bytes','files':len(paths),'python_files':python_files,
            'bytes':total,'maximum_file_bytes':maximum,'problems':problems,'ok':bool(paths) and not problems,
            'limitations':'Finite pattern and size/source checks; not a complete security or license audit.'}
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['ok'] else 1

if __name__=='__main__':raise SystemExit(main())
