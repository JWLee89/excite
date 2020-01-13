"""
    Credits to joeld at stackoverflow for the examples and also the link to the
    blender build script for the bcolors class
    link: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
"""


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_str_builder(color):
    """
        Function to building colored string
        :param color: The color that we want to display.
        :return:
    """
    def colored_str(*input_str, sep=" ", start=None, end=None):
        """
            :param input_str: The input string. Simi
            :param sep: The separator for each of the inputs passed
            :param start: The starting index of the string. If not specified, equals to 0.
            :param end: The ending index of the colored portion. If not specified, equals to
            length of the string.
            :return: Colored version of the string
        """
        concat_str = sep.join(input_str)
        if end is None and start is None:
            return color + concat_str + bcolors.ENDC
        elif start is None:
            start = 0
        elif end is None:
            end = len(concat_str)
        return concat_str[:start] + color + concat_str[start:end] + bcolors.ENDC
    return colored_str


# Maybe this might not be a good idea when working with large strings
warning_str = colored_str_builder(bcolors.WARNING)
info_str = colored_str_builder(bcolors.OKBLUE)
fail_str = colored_str_builder(bcolors.FAIL)
ok_str = colored_str_builder(bcolors.OKGREEN)
bold_str = colored_str_builder(bcolors.BOLD)


if __name__ == "__main__":
    print(warning_str("this is a warning", " this is a test"))
    print(bold_str("this is a warning", start=2))
