"""
    @Author Jay Lee
    Simple interface for interacting with the CLI.
    One concrete implementation thus far is an implementation
    of the Tmux interface. Currently a work in progress.
    Feel free to add more of your favorite commands.
"""

COMMAND_PREFIX = "tmux "


def _append_args(input_str, *fixed_args):
    """
        Append the arguments to the original input string
        :param input:
        :param fixed_args:
        :return:
    """
    for arg in fixed_args:
        input_str += arg
        input_str += " "
    return input_str


def create_command(cmd, *fixed_args):
    """
        :param cmd: The actual command.
        :param fixed_args: Args for adding fixed prefixes to a command
        :return: The tmux command to be executed
    """
    cmd_str = cmd + " "
    cmd_str = _append_args(cmd_str, *fixed_args)

    def command(*args):
        """
            :param args: Additional arguments and flags to be inserted by the user.
            :return:
        """
        return _append_args(cmd_str, *args)

    return command


def create_tmux_command(cmd, *fixed_args, prefix=''):
    """
        :param cmd: The actual command.
        :param fixed_args: Args for adding fixed prefixes to a command
        :return: The tmux command to be executed
    """
    cmd_str = COMMAND_PREFIX + cmd + " "
    cmd_str = _append_args(cmd_str, *fixed_args)

    def command(*args):
        """
            :param args: Additional arguments and flags to be inserted by the user.
            :return:
        """
        return _append_args(cmd_str, *args)

    return command


class Shell:
    """
        Common commands. These won't be used, since there are better and more robust
        python equivalents, but it is here in case people want to use it
    """
    ls = create_command("ls")
    cd = create_command("cd")
    mkdir = create_command("mkdir")


class Tmux:
    """
        Interface containing a list of common tmux comamnds used by me :)
    """
    # Commands
    ls = create_tmux_command("ls")
    new_session = create_tmux_command("new-session", '-s')
    rename_session = create_tmux_command("rename-session")
    kill_session = create_tmux_command("kill-session", '-t')
    kill_current_session = lambda: "exit"
    # detach = create_command("detach")
    # So that we can detach while a script that takes
    # days to run is running
    detach = lambda: chr(2) + chr(ord('d'))     # ctrl-d + b
    attach = create_tmux_command("attach", '-t')


if __name__ == "__main__":
    print(Tmux.ls())
    print(Tmux.new_session("teemo"))
    print(Tmux.kill_session("teemo"))
    print(Tmux.detach())
    print(Tmux.attach('teemo'))
    print(Shell.ls('-l'))
    print(Shell.ls('-l', '-all'))