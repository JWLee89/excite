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

    def colored_str(input_str, start=0, end=None):
        """
            :param input_str:
            :param start:
            :param end:
            :return:
        """
        if end is None:
            end = len(input_str)
        return input_str[:start] + color + input_str[start:end] + bcolors.ENDC
    return colored_str


# Maybe this might not be a good idea when working with large strings
warning_str = colored_str_builder(bcolors.WARNING)
info_str = colored_str_builder(bcolors.OKBLUE)
fail_str = colored_str_builder(bcolors.FAIL)
ok_str = colored_str_builder(bcolors.OKGREEN)
bold_str = colored_str_builder(bcolors.BOLD)


if __name__ == "__main__":
    print(warning_str("this is a warning"))
    print(bold_str("this is a warning", 2))
