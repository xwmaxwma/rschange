# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from argparse import Action, ArgumentParser, Namespace
from collections import OrderedDict, abc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from addict import Dict
from utils.util import check_file_exist
import re

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
DEPRECATION_KEY = '_deprecation_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text', 'env_variables']


class ConfigDict(Dict):
        
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def __copy__(self):
        other = self.__class__()
        for key, value in super().items():
            other[key] = value
        return other

    copy = __copy__


    def merge(self, other: dict):
        """Merge another dictionary into current dictionary.

        Args:
            other (dict): Another dictionary.
        """
        default = object()

        def _merge_a_into_b(a, b):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    a.pop(DELETE_KEY, None)
                    return a
                if a.pop(DELETE_KEY, False):
                    b.clear()
                all_keys = list(b.keys()) + list(a.keys())
                return {
                    key:
                    _merge_a_into_b(a.get(key, default), b.get(key, default))
                    for key in all_keys if key != DELETE_KEY
                }
            else:
                return a if a is not default else b

        merged = _merge_a_into_b(copy.deepcopy(other), copy.deepcopy(self))
        self.clear()
        for key, value in merged.items():
            self[key] = value

class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node

class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml.
    ``Config.fromfile`` can parse a dictionary from a config file, then
    build a ``Config`` instance with the dictionary.
    The interface is the same as a dict object and also allows access config
    values as attributes.

    Args:
        cfg_dict (dict, optional): A config dictionary. Defaults to None.
        cfg_text (str, optional): Text of config. Defaults to None.
        filename (str or Path, optional): Name of config file.
            Defaults to None.
        format_python_code (bool): Whether to format Python code by yapf.
            Defaults to True.

    Here is a simple example:

    Examples:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/username/projects/mmengine/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/username/projects/mmengine/tests/data/config/a.py]
        :"
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    You can find more advance usage in the `config tutorial`_.

    .. _config tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
    """  # noqa: E501

    def __init__(
        self,
        cfg_dict: dict = None,
        cfg_text: Optional[str] = None,
        filename: Optional[Union[str, Path]] = None,
        env_variables: Optional[dict] = None,
        format_python_code: bool = True,
    ):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)
        super().__setattr__('_cfg_dict', cfg_dict)
        super().__setattr__('_filename', filename)
        super().__setattr__('_format_python_code', format_python_code)
        if not hasattr(self, '_imported_names'):
            super().__setattr__('_imported_names', set())

        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__('_env_variables', env_variables)

    @staticmethod
    def fromfile(filename: Union[str, Path],
                 use_predefined_variables: bool = True,
                 import_custom_modules: bool = True,
                 use_environment_variables: bool = True,
                 format_python_code: bool = True) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
    
        cfg_dict, cfg_text, env_variables = Config._file2dict(
            filename, use_predefined_variables, use_environment_variables)
            
        return Config(
            cfg_dict,
            cfg_text=cfg_text,
            filename=filename,
            env_variables=env_variables)


    @staticmethod
    def _get_base_modules(nodes: list) -> list:
        """Get base module name from parsed code.

        Args:
            nodes (list): Parsed code of the config file.

        Returns:
            list: Name of base modules.
        """

        def _get_base_module_from_with(with_nodes: list) -> list:
            """Get base module name from if statement in python file.

            Args:
                with_nodes (list): List of if statement.

            Returns:
                list: Name of base modules.
            """
            base_modules = []
            for node in with_nodes:
                assert isinstance(node, ast.ImportFrom), (
                    'Illegal syntax in config file! Only '
                    '`from ... import ...` could be implemented` in '
                    'with read_base()`')
                assert node.module is not None, (
                    'Illegal syntax in config file! Syntax like '
                    '`from . import xxx` is not allowed in `with read_base()`')
                base_modules.append(node.level * '.' + node.module)
            return base_modules

        for idx, node in enumerate(nodes):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == BASE_KEY):
                raise RuntimeError(
                    'The configuration file type in the inheritance chain '
                    'must match the current configuration file type, either '
                    '"lazy_import" or non-"lazy_import". You got this error '
                    f'since you use the syntax like `_base_ = "{node.targets[0].id}"` '  # noqa: E501
                    'in your config. You should use `with read_base(): ... to` '  # noqa: E501
                    'mark the inherited config file. See more information '
                    'in https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html'  # noqa: E501
                )

            if not isinstance(node, ast.With):
                continue

            expr = node.items[0].context_expr
            if (not isinstance(expr, ast.Call)
                    or not expr.func.id == 'read_base' or  # type: ignore
                    len(node.items) > 1):
                raise RuntimeError(
                    'Only `read_base` context manager can be used in the '
                    'config')

            # The original code:
            # ```
            # with read_base():
            #     from .._base_.default_runtime import *
            # ```
            # The processed code:
            # ```
            # from .._base_.default_runtime import *
            # ```
            # As you can see, the if statement is removed and the
            # from ... import statement will be unindent
            for nested_idx, nested_node in enumerate(node.body):
                nodes.insert(idx + nested_idx + 1, nested_node)
            nodes.pop(idx)
            return _get_base_module_from_with(node.body)
        return []

    @staticmethod
    def _validate_py_syntax(filename: str):
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        """Substitute predefined variables in config with actual values.
        """
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _substitute_env_variables(filename: str, temp_config_name: str):
        """Substitute environment variables in config with actual values.

        Sometimes, we want to change some items in the config with environment
        variables. For examples, we expect to change dataset root by setting
        ``DATASET_ROOT=/dataset/root/path`` in the command line. This can be
        easily achieved by writing lines in the config as follows

        .. code-block:: python

           data_root = '{{$DATASET_ROOT:/default/dataset}}/images'


        Here, ``{{$DATASET_ROOT:/default/dataset}}`` indicates using the
        environment variable ``DATASET_ROOT`` to replace the part between
        ``{{}}``. If the ``DATASET_ROOT`` is not set, the default value
        ``/default/dataset`` will be used.

        Environment variables not only can replace items in the string, they
        can also substitute other types of data in config. In this situation,
        we can write the config as below

        .. code-block:: python

           model = dict(
               bbox_head = dict(num_classes={{'$NUM_CLASSES:80'}}))


        For details, Please refer to docs/zh_cn/tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        regexp = r'\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}'
        keys = re.findall(regexp, config_file)
        env_variables = dict()
        for var_name, value in keys:
            regexp = r'\{\{[\'\"]?\s*\$' + var_name + r'\s*\:\s*' \
                + value + r'\s*[\'\"]?\}\}'
            if var_name in os.environ:
                value = os.environ[var_name]
                env_variables[var_name] = value
            if not value:
                raise KeyError(f'`{var_name}` cannot be found in `os.environ`.'
                               f' Please set `{var_name}` in environment or '
                               'give a default value.')
            config_file = re.sub(regexp, value, config_file)

        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        return env_variables

    @staticmethod
    def _pre_substitute_base_vars(filename: str,
                                  temp_config_name: str) -> dict:
        """Preceding step for substituting variables in base config with actual
        value.

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.

        Returns:
            dict: A dictionary contains variables in base config.
        """
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r'\{\{\s*' + BASE_KEY + r'\.([\w\.]+)\s*\}\}'
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f'_{base_var}_{uuid.uuid4().hex.lower()[:6]}'
            base_var_dict[randstr] = base_var
            regexp = r'\{\{\s*' + BASE_KEY + r'\.' + base_var + r'\s*\}\}'
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg: Any, base_var_dict: dict,
                              base_cfg: dict) -> Any:
        """Substitute base variables from strings to their actual values.

        Args:
            Any : Config dictionary.
            base_var_dict (dict): A dictionary contains variables in base
                config.
            base_cfg (dict): Base config dictionary.

        Returns:
            Any : A dictionary with origin base variables
                substituted with actual values.
        """
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split('.'):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(
                        v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split('.'):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(
            filename: str,
            use_predefined_variables: bool = True,
            use_environment_variables: bool = True) -> Tuple[dict, str, dict]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.

        Returns:
            Tuple[dict, str]: Variables dictionary and text of Config.
        """
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py']:
            raise OSError('Only py type are supported now!')
        try:
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix=fileExtname, delete=False)
                if platform.system() == 'Windows':
                    temp_config_file.close()

                # Substitute predefined variables
                if use_predefined_variables:
                    Config._substitute_predefined_vars(filename,
                                                       temp_config_file.name)
                else:
                    shutil.copyfile(filename, temp_config_file.name)
                # Substitute environment variables
                env_variables = dict()
                if use_environment_variables:
                    env_variables = Config._substitute_env_variables(
                        temp_config_file.name, temp_config_file.name)
                # Substitute base variables from placeholders to strings
                base_var_dict = Config._pre_substitute_base_vars(
                    temp_config_file.name, temp_config_file.name)

                # Handle base files
                base_cfg_dict = ConfigDict()
                cfg_text_list = list()
                for base_cfg_path in Config._get_base_files(
                        temp_config_file.name):
                    base_cfg_path, scope = Config._get_cfg_path(
                        base_cfg_path, filename)
                    _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                        filename=base_cfg_path,
                        use_predefined_variables=use_predefined_variables,
                        use_environment_variables=use_environment_variables)
                    cfg_text_list.append(_cfg_text)
                    env_variables.update(_env_variables)
                    duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                    if len(duplicate_keys) > 0:
                        raise KeyError(
                            'Duplicate key is not allowed among bases. '
                            f'Duplicate keys: {duplicate_keys}')

                    # _dict_to_config_dict will do the following things:
                    # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                    # 2. Set `_scope_` for the outer dict variable for the base
                    # config.
                    # 3. Set `scope` attribute for each base variable.
                    # Different from `_scope_`， `scope` is not a key of base
                    # dict, `scope` attribute will be parsed to key `_scope_`
                    # by function `_parse_scope` only if the base variable is
                    # accessed by the current config.
                    _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                    base_cfg_dict.update(_cfg_dict)

                with open(temp_config_file.name, encoding='utf-8') as f:
                    parsed_codes = ast.parse(f.read())
                    parsed_codes = RemoveAssignFromAST(BASE_KEY).visit(
                        parsed_codes)
                codeobj = compile(parsed_codes, '', mode='exec')
                # Support load global variable in nested function of the
                # config.
                global_locals_var = {BASE_KEY: base_cfg_dict}
                ori_keys = set(global_locals_var.keys())
                eval(codeobj, global_locals_var, global_locals_var)
                cfg_dict = {
                    key: value
                    for key, value in global_locals_var.items()
                    if (key not in ori_keys and not key.startswith('__'))
                }
                # close temp file
                for key, value in list(cfg_dict.items()):
                    if isinstance(value,
                                  (types.FunctionType, types.ModuleType)):
                        cfg_dict.pop(key)
                temp_config_file.close()

                # If the current config accesses a base variable of base
                # configs, The ``scope`` attribute of corresponding variable
                # will be converted to the `_scope_`.
                Config._parse_scope(cfg_dict)
        except Exception as e:
            if osp.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            raise e


        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {
            k: v
            for k, v in cfg_dict.items() if not k.startswith('__')
        }

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables

    @staticmethod
    def _dict_to_config_dict_lazy(cfg: dict):
        """Recursively converts ``dict`` to :obj:`ConfigDict`. The only
        difference between ``_dict_to_config_dict_lazy`` and
        ``_dict_to_config_dict_lazy`` is that the former one does not consider
        the scope, and will not trigger the building of ``LazyObject``.

        Args:
            cfg (dict): Config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            cfg_dict = ConfigDict()
            for key, value in cfg.items():
                cfg_dict[key] = Config._dict_to_config_dict_lazy(value)
            return cfg_dict
        if isinstance(cfg, (tuple, list)):
            return type(cfg)(
                Config._dict_to_config_dict_lazy(_cfg) for _cfg in cfg)
        return cfg

    @staticmethod
    def _dict_to_config_dict(cfg: dict,
                             scope: Optional[str] = None,
                             has_scope=True):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.
            scope (str, optional): Scope of instance.
            has_scope (bool): Whether to add `_scope_` key to config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            if has_scope and 'type' in cfg:
                has_scope = False
                if scope is not None and cfg.get('_scope_', None) is None:
                    cfg._scope_ = scope  # type: ignore
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, 'scope', scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(
                    value, scope=scope, has_scope=has_scope)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            ]
        return cfg

    @staticmethod
    def _parse_scope(cfg: dict) -> None:
        """Adds ``_scope_`` to :obj:`ConfigDict` instance, which means a base
        variable.

        If the config dict already has the scope, scope will not be
        overwritten.

        Args:
            cfg (dict): Config needs to be parsed with scope.
        """
        if isinstance(cfg, ConfigDict):
            cfg._scope_ = cfg.scope
        elif isinstance(cfg, (tuple, list)):
            [Config._parse_scope(value) for value in cfg]
        else:
            return

    @staticmethod
    def _get_base_files(filename: str) -> list:
        """Get the base config file.

        Args:
            filename (str): The config file.

        Raises:
            TypeError: Name of config file.

        Returns:
            list: A list of base config.
        """
        file_format = osp.splitext(filename)[1]
        if file_format == '.py':
            Config._validate_py_syntax(filename)
            with open(filename, encoding='utf-8') as f:
                parsed_codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (isinstance(c, ast.Assign)
                            and isinstance(c.targets[0], ast.Name)
                            and c.targets[0].id == BASE_KEY)

                base_code = next((c for c in parsed_codes if is_base_line(c)),
                                 None)
                if base_code is not None:
                    base_code = ast.Expression(  # type: ignore
                        body=base_code.value)  # type: ignore
                    base_files = eval(compile(base_code, '', mode='eval'))
                else:
                    base_files = []
        else:
            raise TypeError('The config type should be py, but got {file_format}')
        base_files = base_files if isinstance(base_files,
                                              list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str,
                      filename: str) -> Tuple[str, Optional[str]]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            Tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        cfg_dir = osp.dirname(filename)
        cfg_path = osp.join(cfg_dir, cfg_path)
        return cfg_path, None

    @staticmethod
    def _merge_a_into_b(a: dict,
                        b: dict,
                        allow_list_keys: bool = False) -> dict:
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @property
    def filename(self) -> str:
        """get file name of config."""
        return self._filename

    @property
    def text(self) -> str:
        """get config text."""
        return self._text

    @property
    def env_variables(self) -> dict:
        """get used environment variables."""
        return self._env_variables

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list_tuple(k, v, use_mapping=False):
            if isinstance(v, list):
                left = '['
                right = ']'
            else:
                left = '('
                right = ')'

            v_str = f'{left}\n'
            # check if all items in the list are dict
            for item in v:
                if isinstance(item, dict):
                    v_str += f'dict({_indent(_format_dict(item), indent)}),\n'
                elif isinstance(item, tuple):
                    v_str += f'{_indent(_format_list_tuple(None, item), indent)},\n'  # noqa: 501
                elif isinstance(item, list):
                    v_str += f'{_indent(_format_list_tuple(None, item), indent)},\n'  # noqa: 501
                elif isinstance(item, str):
                    v_str += f'{_indent(repr(item), indent)},\n'
                else:
                    v_str += str(item) + ',\n'
            if k is None:
                return _indent(v_str, indent) + right
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent) + right
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _format_list_tuple(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict]:
        return (self._cfg_dict, self._filename, self._text,
                self._env_variables)

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        super(Config, other).__setattr__('_cfg_dict', self._cfg_dict.copy())

        return other

    copy = __copy__

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str],
                                        dict]):
        _cfg_dict, _filename, _text, _env_variables = state
        super().__setattr__('_cfg_dict', _cfg_dict)
        super().__setattr__('_filename', _filename)
        super().__setattr__('_text', _text)
        super().__setattr__('_text', _env_variables)

    def merge_from_dict(self,
                        options: dict,
                        allow_list_keys: bool = True) -> None:
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
                are allowed in ``options`` and will replace the element of the
                corresponding index in the config if the config is a list.
                Defaults to True.

        Examples:
            >>> from mmengine import Config
            >>> #  Merge dictionary element
            >>> options = {'model.backbone.depth': 50, 'model.backbone.with_cp': True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg._cfg_dict
            {'model': {'backbone': {'type': 'ResNet', 'depth': 50, 'with_cp': True}}}
            >>> # Merge list element
            >>> cfg = Config(
            >>>     dict(pipeline=[dict(type='LoadImage'),
            >>>                    dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg._cfg_dict
            {'pipeline': [{'type': 'SelfLoadImage'}, {'type': 'LoadAnnotations'}]}
        """  # noqa: E501
        option_cfg_dict: dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__('_cfg_dict')
        super().__setattr__(
            '_cfg_dict',
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))

    def _to_lazy_dict(self, keep_imported: bool = False) -> dict:
        """Convert config object to dictionary and filter the imported
        object."""
        res = self._cfg_dict._to_lazy_dict()
        if hasattr(self, '_imported_names') and not keep_imported:
            res = {
                key: value
                for key, value in res.items()
                if key not in self._imported_names
            }
        return res

    def to_dict(self, keep_imported: bool = False):
        """Convert all data in the config to a builtin ``dict``.

        Args:
            keep_imported (bool): Whether to keep the imported field.
                Defaults to False

        If you import third-party objects in the config file, all imported
        objects will be converted to a string like ``torch.optim.SGD``
        """
        return self._cfg_dict.to_dict()

if __name__ == "__main__":
    file_path = "/train-syncdata/xiaowen.ma/mycode/rssegmentation/configs/ssnet.py"
    cfg = Config.fromfile(file_path)
    print(cfg.model_config)