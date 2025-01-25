from typing import Dict, Any

import lark

from base import AbstractTransform

class PromptSequenceTransform(AbstractTransform):
    def __init__(self, target: dict, args: dict):
        super().__init__(args)
        self.target = target
        self.parser = lark.Lark(r"""
        !start: (prompt | /[][():]/+)*
        prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
        !emphasized: "(" prompt ")"
                | "(" prompt ":" prompt ")"
                | "[" prompt "]"
        scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
        alternate: "[" prompt ("|" prompt)+ "]"
        WHITESPACE: /\s+/
        plain: /([^\\\[\]():|]|\\.)+/
        %import common.SIGNED_NUMBER -> NUMBER
        """)

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target
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
        return cls(json["target"], json["args"]) 
        
    def apply(self, source, steps=1, verbose=False): 
        sequence = self.get_prompt_sequence  
        for s in range(max(1,steps)):
            if verbose:
                print(f"[{s+1}/{steps}] {(s+1)/steps}%")
            params = self.lerp_params(self.args, (s+1)/steps, verbose=verbose)
            config = self.step(source, self.target, params, 
                                        verbose=verbose)
            self.step_results.append(config)
            yield config

    def step(self, source, target, params, verbose=False):
        raise NotImplementedError

    def get_prompt_sequence(self, prompts, steps):
        """
        >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
        >>> g("test")
        [[10, 'test']]
        >>> g("a [b:3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [b: 3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [[[b]]:2]")
        [[2, 'a '], [10, 'a [[b]]']]
        >>> g("[(a:2):3]")
        [[3, ''], [10, '(a:2)']]
        >>> g("a [b : c : 1] d")
        [[1, 'a b  d'], [10, 'a  c  d']]
        >>> g("a[b:[c:d:2]:1]e")
        [[1, 'abe'], [2, 'ace'], [10, 'ade']]
        >>> g("a [unbalanced")
        [[10, 'a [unbalanced']]
        >>> g("a [b:.5] c")
        [[5, 'a  c'], [10, 'a b c']]
        >>> g("a [{b|d{:.5] c")  # not handling this right now
        [[5, 'a  c'], [10, 'a {b|d{ c']]
        >>> g("((a][:b:c [d:3]")
        [[3, '((a][:b:c '], [10, '((a][:b:c d']]
        """

        def collect_steps(steps, tree):
            l = [steps]
            class CollectSteps(lark.Visitor):
                def scheduled(self, tree):
                    tree.children[-1] = float(tree.children[-1])
                    if tree.children[-1] < 1:
                        tree.children[-1] *= steps
                    tree.children[-1] = min(steps, int(tree.children[-1]))
                    l.append(tree.children[-1])
                def alternate(self, tree):
                    l.extend(range(1, steps+1))
            CollectSteps().visit(tree)
            return sorted(set(l))

        def at_step(step, tree):
            class AtStep(lark.Transformer):
                def scheduled(self, args):
                    before, after, _, when = args
                    yield before or () if step <= when else after
                def alternate(self, args):
                    yield next(args[(step - 1)%len(args)])
                def start(self, args):
                    def flatten(x):
                        if type(x) == str:
                            yield x
                        else:
                            for gen in x:
                                yield from flatten(gen)
                    return ''.join(flatten(args))
                def plain(self, args):
                    yield args[0].value
                def __default__(self, data, children, meta):
                    for child in children:
                        yield from child
            return AtStep().transform(tree)

        def get_schedule(prompt):
            try:
                tree = self.parser.parse(prompt)
            except lark.exceptions.LarkError as e:
                if 0:
                    import traceback
                    traceback.print_exc()
                return [[steps, prompt]]
            return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

        promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
        return [promptdict[prompt] for prompt in prompts]    
    
