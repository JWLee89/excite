"""
    @Author Jay Lee
    A ssh session manager that provides an interface over paramiko.
    Something that I plan on using to speed up the experimentation process during
    my research and life as a grad student.

    This is especially useful for re-starting experiments if in worst-cases where
    experiments will stop halfway due to a power outage, or server overload

    Some features include
    - Programatically creating tmux sessions
    - Automating the running of experiments over the multiple gpu-based servers

    Dependencies
    - Paramiko: (link: https://github.com/paramiko/paramiko)
"""
import paramiko


def _do_print(*items):
    """
        Simple function for handling printing.
        In case we want to change behavior in the future to logging.
        :param items:
        :return:
    """
    print("DEBUG::: ", *items)


def isiterable(target):
    """
        From the pysistant library (@author Jay Lee)
        Check if target object is iterable
        :param target:
        :return: true if target is iterable. Otherwise, return false
    """
    try:
        iter(target)
    except:
        return False
    else:
        return True


def create_connections(connections, username, password, port=9999):
    """

        :param connections:
        :param username:
        :param password:
        :param port:
        :return:
    """
    result = []
    for url in connections:
        client_name = url.split(".")[0]
        print(f"URL: {url}. Client name: {client_name}")
        result.append(SshConnection(url, username, password, client_name=client_name, port=port, is_debug=True))
    return result


class SshConnectionManager:
    """
        Class for managing multiple ssh connections
    """

    def __init__(self, connections):
        self.connections = connections

    def connect(self, client):
        """
            Connect to sa specific client
            :param client: Can access by index
            :return:
        """
        if type(int):
            if client >= len(self.connections):
                raise IndexError(f"Index {client} out of range. "
                                 f"Connection count: {len(self.connections)}")
            self.connections[client].open()
        elif type(str):
            for connection in self.connections:
                if connection.name == client:
                    connection.open()

    def gpu_info(self):
        result = {}
        for connection in self.connections:
            result[connection.client_name] = connection.gpu_info()
        return result

    def __getitem__(self, position):
        return self.connections[position]

    def __len__(self):
        return len(self.connections)

    def __repr__(self):
        return f"<class>SshConnectionManager. Clients: {self.connections}"


def parse_nvidia_smi_output(nvidia_smi_output):
    """
        TODO
        Parse the output of "nvidia-smi" command to get gpu info
        :param nvidia_smi_output: the output of the "nvidia-smi" command
        :return:
    """
    result = []


class GPU:
    def __init__(self):
        pass


class SshConnection:
    """
        An SshConnection object represents a connection between a remote machine and the
        local host.
    """
    def __init__(self, url, username, password, client_name, port=22, is_debug=False):
        """
            :param url:
            :param username:
            :param password:
            :param client_name:
            :param port: The
            :param is_debug:
        """
        self.url = url
        self.username = username
        self.password = password
        self.client_name = client_name
        self.port = port
        self.is_debug = is_debug
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def open(self):
        """
            Access remote or local server via ssh
            :return:
        """
        if self._isalive():
            if self.is_debug:
                _do_print("connection is still alive!")
            return
        if self.is_debug:
            _do_print(f"Attempting the following command: ssh {self.username}@{self.url} -p {self.port}")
        self.ssh.connect(self.url, self.port, self.username, self.password)

    def _isalive(self):
        """
            Check if connection is alive
            :return:
        """
        transport_is_not_none = self.ssh.get_transport() is not None
        return transport_is_not_none and self.ssh.get_transport().is_active()

    def close(self):
        """
            Close connection if it is alive
            :return:
        """
        if self._isalive():
            self.ssh.close()

    def cmd(self, shell_command, close_after=False, do_on_finished=None):
        """
            :param shell_command: Command can either be string or an iterable set of commands
            :param close_after: If set to true, close the connection after running commands
            :param do_on_finished:
            :return:
        """
        self.open()
        # Pass command
        if isiterable(shell_command):
            output = []
            for command in shell_command:
                command_output = self._execute_cmd(command)
                output.append(command_output)
                if self.is_debug:
                    _do_print(f"Command executed. Output: {command_output}")
        elif type(str):
            output = self._execute_cmd(shell_command)
        else:
            raise TypeError(f"Argument should be a string literal command "
                            f"or a set of commands. Problem with: {shell_command}")

        if close_after:
            self.close()

        # Run additional logic such as sending an email to me once the
        # experiments have finished running
        if hasattr(do_on_finished, "__call__"):
            do_on_finished()

        return output

    def create_tmux_session(self, session_name):
        pass

    def _execute_cmd(self, command):
        """

            :param targets:
            :return:
        """
        stdin, stdout, stderr = self.ssh.exec_command(command)
        outlines = stdout.readlines()
        resp = ''.join(outlines)
        return resp

    def gpu_info(self):
        """
            :return: Returns a list of GPU information via the "nvidia-smi" command
        """
        result = []
        nvidia_smi_output = self.cmd("nvidia-smi")
        if self.is_debug:
            _do_print(f"------------------------------ {self.client_name} ------------------------------  \n{nvidia_smi_output}")
        result.append(nvidia_smi_output)
        return result

    def __repr__(self):
        return f"<SshConnection>: [url={self.url}, username={self.username}, client_name={self.client_name}" \
               f", port={self.port}]"