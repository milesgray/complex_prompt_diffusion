import os, json
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Union, Dict, Any

from cpd.util import from_json
from cpd.embeddings.prompts import ComplexPrompt


class AbstractTransform:
    def __init__(self, args: dict):
        self.args = args
        self.param_lerp_keys = args['lerp_keys'] if 'lerp_keys' in args else []
        self.step_results = []

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json(self) -> Dict[str, Any]:
        """
        Packs this instance to a JSON dictionary that can be serialized.
        All values are strings or containers with strings.

        Returns:
            `dict`: Serializable dictionary containing all the attributes that make up this instance.
        """
        type_str = str(type(self)).replace("<class '","").replace("'>","")
        module_str = ".".join(type_str.split(".")[:-1])
        json = {        
            "args": self.args,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
        }      
        return json

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """        
        return json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(json["args"]) 

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs):
        json = cls._dict_from_json_file(json_file)
        return cls.from_json(json, **kwargs)

class AbstractPromptTransform(AbstractTransform):
    def __init__(self, target: ComplexPrompt, args: dict):
        super().__init__(args)
        self.target = target
        self.cache = OrderedDict()
        
    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target.to_json()
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(from_json(json["target"], **kwargs), json["args"]) 
        
    def apply(self, prompt_start, steps=1, verbose=False): 
        if len(self.param_lerp_keys) == 0 or \
            all([k not in self.args for k in self.param_lerp_keys]):
            steps = 1
            if verbose:
                print(f"No valid interpolatable parameters found. Set them as values for 'lerp_keys' in config dictionary")    
        for s in range(max(1,steps)):
            if verbose:
                print(f"[{s+1}/{steps}] {(s+1)/steps}%")
            params = self.lerp_params(self.args, (s+1)/steps, verbose=verbose)
            batch_embedding = self.step(prompt_start, self.target, params, 
                                        verbose=verbose)
            self.step_results.append(batch_embedding)
        return self.step_results

    def step(self, prompt_start, prompt_end, params, verbose=False):
        raise NotImplementedError

    def lerp_params(self, params, amount, verbose=False):
        if amount == 1:
            return params
        result = {}        
        for k,v in params.items():
            if k not in self.param_lerp_keys:
                result[k] = v
            else:
                if isinstance(v, float):
                    result[k] = v * amount
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                elif isinstance(v, int):
                    result[k] = int(v * amount)
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                elif isinstance(v, tuple):
                    if len(v) != 2: 
                        result[k] = v
                    elif isinstance(v[0], int) and isinstance(v[1], int):
                        v[0] = int(v[0] * amount)
                        v[1] = int(v[1] + v[1] * (1-amount))
                        result[k] = v
                    elif isinstance(v[0], float) and isinstance(v[1], float):
                        v[0] = v[0] * amount
                        v[1] = v[1] + v[1] * (1-amount)
                        result[k] = v
                    else:
                        result[k] = v
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                else:
                    result[k] = v
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
        if verbose:
            print(f"Result: {result}")
        return result
