import importlib.util
for p in ['fitz','pypdf','pip']:
    print(p, importlib.util.find_spec(p))
