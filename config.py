import yaml

class Config:
    def __init__(self, dictionary):
        self._dict = dictionary
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)
        
    def to_dict(self):
        return self._dict
