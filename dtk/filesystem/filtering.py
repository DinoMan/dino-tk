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
    def __init__(self, regex, start=0, end=None, criterion=None):
        if criterion is None:
            self.criterion = Filter()

        self.regex = re.compile(regex)
        self.start = start
        self.end = end

    def __call__(self, string):
        if self.criterion(string):
            match_limits = self.regex.search(string)
            if self.end is None:
                return string[match_limits.start():match_limits.end() + 1][self.start:]
            else:
                return string[match_limits.start():match_limits.end() + 1][self.start:self.end]
        else:
            return ""


class ExtensionMatcher():
    def __init__(self, ext):
        if 'basestring' not in globals():
            basestring = str

        if isinstance(ext, basestring):
            self.extensions = [ext]
        else:
            self.extensions = ext

    def __call__(self, string):
        extension = os.path.splitext(string)[1]
        return extension in self.extensions


class Filter():
    def __init__(self, is_file=False, ext=None, regex=[]):
        self.rules = []
        self.negative_rules = []

        if ext is not None:
            self.rules.append(ExtensionMatcher(ext))

        if is_file:
            self.rules.append(os.path.isfile)

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

        return new_filter

    def __add__(self, other):
        new_filter = Filter()
        new_filter.rules += self.rules
        new_filter.rules += other.rules
        new_filter.negative_rules += self.negative_rules
        new_filter.negative_rules += other.negative_rules

        return new_filter

    def __add_regex__(self, regex):
        regex = re.compile(regex)
        self.rules.append(regex.search)
