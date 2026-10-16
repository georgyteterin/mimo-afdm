import yaml

def get_tests(path):
    with open(path, encoding="utf-8") as f:
        curves = yaml.safe_load(f)["curves"]
    return [{"name": name, **cfg} for name, cfg in curves.items()]