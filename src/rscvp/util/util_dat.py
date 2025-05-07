__all__ = ['pattern_sort']


def pattern_sort(string_list: list[str], pattern_list: list[str]) -> list[str]:
    """
    sort string list based on pattern list.
    pattern not found in the string list, then put at the end

    >>> ls = ['it_a', 'it_b', 'it_c']
    ... print(pattern_sort(ls, ['c', 'a', 'b']))
    ['it_c', 'it_a', 'it_b']

    :param string_list:
    :param pattern_list:
    :return: sorted string list
    """

    def pattern_priority(s):
        for idx, pattern in enumerate(pattern_list):
            if pattern in s:
                return idx
            else:
                return len(pattern_list)  # no pattern is found, put at the end

    return sorted(string_list, key=pattern_priority)
