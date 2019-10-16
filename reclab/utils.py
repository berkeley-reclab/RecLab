import collections


def convert_to_iterable(x):
    """Convert x to an iterable if it isn't already one."""
    if not isinstance(x, collections.Iterable):
        return [x]
    else:
        return x

