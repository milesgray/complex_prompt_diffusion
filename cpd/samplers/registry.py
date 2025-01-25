import copy

lookup = {}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(spec, args=None):
    if args is not None:
        spec_args = copy.deepcopy(spec['args'])
        spec_args.update(args)
    else:
        spec_args = spec['args']
    result = lookup[spec['name']](name=spec['name'], **spec_args)
    return result

def create(name, **kwargs):
    if isinstance(name, str):
        if name in lookup:
            return make({"name": name, "args": kwargs})
        else:
            if len(kwargs):
                return eval(name)(**kwargs)
            else:
                return eval(name)
    raise ValueError(f"Must pass name of registered component or valid python code to `create` methods:\n'{name}'\nis invalid")