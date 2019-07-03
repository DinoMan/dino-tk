import re
import os.path


class Tagger():
    def __init__(self, extractor, translation_dict=None):
        self.extractor = extractor
        self.translation_dict = translation_dict

    def __call__(self, string):
        if self.translation_dict is None:
            return self.extractor(string)
        else:
            extracted_pattern = self.extractor(string)
            if extracted_pattern is not None:
                return self.translation_dict[extracted_pattern]
            else:
                return "N/A"


class PatternExtractor():
    def __init__(self, regex, start=0, end=-1, criterion=None):
        if criterion is None:
            self.criterion = Filter()

        self.regex = re.compile(regex)
        self.start = start
        self.end = end

    def __call__(self, string):
        if self.criterion(string):
            match_limits = self.regex.search(string)
            return string[match_limits.start():match_limits.end() + 1][self.start:self.end]
        else:
            return ""


class ExtensionMatcher():
    def __init__(self, extension):
        self.extension = extension

    def __call__(self, string):
        return os.path.splitext(string)[1] == self.extension


class Filter():
    def __init__(self, is_file=False, extension=None, regex=[]):
        self.rules = []
        self.negative_rules = []

        if extension is not None:
            self.__has_extension__(extension)

        if is_file:
            self.__is_file__()

        for r in regex:
            self.__add_regex__(r)

    def __call__(self, string):
        match = True
        for rule in self.rules:
            match = bool(rule(string)) and match
        for rule in self.negative_rules:
            match = not bool(rule(string)) and match

        return match

    def __invert__(self):
        new_filter = Filter()
        new_filter.rules = self.negative_rules
        new_filter.negative_rules = self.rules
        new_filter.extension = self.extension

        return new_filter

    def __add__(self, other):
        new_filter = Filter()
        new_filter.rules += self.rules
        new_filter.rules += other.rules
        new_filter.negative_rules += self.negative_rules
        new_filter.negative_rules += other.negative_rules

        if self.extension is not None:
            new_filter.extension = self.extension
        elif other.extension is not None:
            new_filter.extension = other.extension
        return new_filter

    def __add_regex__(self, regex):
        regex = re.compile(regex)
        self.rules.append(regex.search)

    def __is_file__(self):
        self.rules.append(os.path.isfile)

    def __has_extension__(self, ext):
        self.rules.append(ExtensionMatcher(ext))

    def __matches_extension__(self, string):
        return os.path.splitext(string)[1] == self.extension
