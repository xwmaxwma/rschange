# Code simplified from https://github.com/open-mmlab/mmengine/blob/main/mmengine/registry/registry.py
# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Callable
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from rich.table import Table
from rich.console import Console


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.
    """

    def __init__(self,
                 name: str,
                 build_func: Optional[Callable] = None):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._imported = False
    
        self.build_func: Callable
        if build_func is None:
            self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        
    def __len__(self):
        return len(self._module_dict)
    
    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance.
        Build an instance by calling :attr:`build_func`.
        """
        return self.build_func(cfg, *args, **kwargs, registry=self)
    
    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            module: Optional[Type] = None) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        """

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                'name must be None, an instance of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name)
            return module

        return _register


def build_from_cfg(
        cfg: dict,
        registry: Registry,
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.
    """
    # Avoid circular import

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
            f'but got {type(default_args)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at '
                    'https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        try:
            # If `obj_cls` inherits from `ManagerMixin`, it should be
            # instantiated by `ManagerMixin.get_instance` to ensure that it
            # can be accessed globally.
            if inspect.isclass(obj_cls) and \
                    issubclass(obj_cls, ManagerMixin):  # type: ignore
                obj = obj_cls.get_instance(**args)  # type: ignore
            else:
                obj = obj_cls(**args)  # type: ignore

            if (inspect.isclass(obj_cls) or inspect.isfunction(obj_cls)
                    or inspect.ismethod(obj_cls)):
                print_log(
                    f'An `{obj_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
                    'registry, and its implementation can be found in '
                    f'{obj_cls.__module__}',  # type: ignore
                    logger='current',
                    level=logging.DEBUG)
            else:
                print_log(
                    'An instance is built from registry, and its constructor '
                    f'is {obj_cls}',
                    logger='current',
                    level=logging.DEBUG)
            return obj

        except Exception as e:
            # Normal TypeError does not print class name.
            cls_location = '/'.join(
                obj_cls.__module__.split('.'))  # type: ignore
            raise type(e)(
                f'class `{obj_cls.__name__}` in '  # type: ignore
                f'{cls_location}.py: {e}')
