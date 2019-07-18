from functools import reduce
import operator


def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def group_factors(factors, optimal_factor=4):
    groups = []
    group = []
    for f in factors:
        group += [f]
        factor = reduce(operator.mul, group, 1)
        if factor > optimal_factor:
            last_factor = group.pop()
            groups.append(reduce(operator.mul, group, 1))
            group = [last_factor]

    groups.append(reduce(operator.mul, group, 1))
    return sorted(groups)