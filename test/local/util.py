__all__ = ['check_attr']


def check_attr(c: type, p: type):
    attr_c = set(dir(c))
    attr_p = set(dir(p))
    if attr_c != attr_p:
        raise RuntimeError(f'{attr_c - attr_p}')
