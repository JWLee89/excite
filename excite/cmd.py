"""
    @Author Jay Lee
    Interface for the Tmux session.
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


def create_tmux_command(cmd, *fixed_args):
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


class CMD:
    """
        Commands to pass
    """


class Tmux:
    """
        Interface containing a list of common tmux comamnds used by me :)
    """
    # Commands
    ls = create_tmux_command("ls")
    new_session = create_tmux_command("new-session", '-s')
    rename_session = create_tmux_command("rename-session")
    kill_session = create_tmux_command("kill-session", '-t')
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