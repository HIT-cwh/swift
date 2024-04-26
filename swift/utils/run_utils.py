from datetime import datetime
from typing import Callable, List, Type, TypeVar, Union

from .logger import get_logger
from .utils import parse_args
import os

logger = get_logger()
_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')
NoneType = type(None)
from xtuner.parallel.sequence import init_dist

def get_main(
    args_class: Type[_TArgsClass], llm_x: Callable[[_TArgsClass], _T]
) -> Callable[[Union[List[str], _TArgsClass, NoneType]], _T]:

    def x_main(argv: Union[List[str], _TArgsClass, NoneType] = None,
               **kwargs) -> _T:
        port = int(os.environ.get('MASTER_PORT', 29500))
        init_dist('slurm', 'nccl', init_backend='deepspeed', port=port)
        os.environ['LOCAL_WORLD_SIZE'] = str(min(int(os.environ['WORLD_SIZE']), 8))
        logger.info(
            f'Start time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}'
        )
        if not isinstance(argv, (list, tuple, NoneType)):
            args, remaining_argv = argv, []
        else:
            args, remaining_argv = parse_args(args_class, argv)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        result = llm_x(args, **kwargs)
        logger.info(
            f'End time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}'
        )
        return result

    return x_main
