import json
from typing import Any, List
from numpyencoder import NumpyEncoder

def json2jl(source: str, target: str) -> None:
    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(target, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def load_jl(path: str) -> List[Any]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data

def save_jl(data: List[Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def save_jl_append(data: str, path: str) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, cls= NumpyEncoder) + '\n')
            
def jl2json(source: str, target: str) -> None:
    data = load_jl(source)
    with open(target, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
        
def save_json(data: str, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def save_json_append(data: str, path: str) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def save_txt(data: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(data)