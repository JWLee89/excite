"""
    @Author Jay Lee
    Interface for the Tmux session.
"""

class Commands:
    """
        Command interface
    """
    PREFIX = "tmux "

    @staticmethod
    def get_command(self, cmd, *args):
        """

            :param self:
            :param cmd: The actual command.
            :param args: Additional arguments and flags to be inserted by the user.
            :return: The tmux command to be executed
        """
        command = self.PREFIX + cmd
        for arg in args:
            command += arg
        return command


class Tmux:

    def __init__(self, SshSession):
        self.session = SshSession

    def cmd(self, cmd_type, *args):
        pass