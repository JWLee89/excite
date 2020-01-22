"""
    @Author Jay Lee
    Utilities go right here.
"""
import time
import functools

DEFAULT_CLOCK_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'


def clock(fmt=DEFAULT_CLOCK_FMT, print_handler=print):
    """
        @Credits to "Fluent Python" chapter on decorators for this neat function.
        :param fmt:
        :param print_handler:
        :return:
    """
    def decorate(func):
        @functools.wraps(func)
        def clocked(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            args_list = []
            # Add args to debug list
            if args:
                args_list.append(', '.join(repr(arg) for arg in args))
            if kwargs:
                pairs = [f'{key}={value}' for key, value in sorted(kwargs.items())]
                args_list.append(', '.join(pairs))
            # Unpack all args and kwargs
            print_handler(fmt.format(**locals()))
            return result
        return clocked
    return decorate