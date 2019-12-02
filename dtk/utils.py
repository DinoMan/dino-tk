from functools import reduce
import operator


def args2dict(args):
    return vars(args)


def dict2args(dictionary):
    arg_list = []
    for k in dictionary.keys():
        if type(dictionary[k]) == type(True):
            if dictionary[k] == True:
                arg_list += ["--" + k]
            else:
                continue
        elif dictionary[k] is None:
            continue

        arg_list += ["--" + k, str(dictionary[k])]
    return arg_list


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


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
