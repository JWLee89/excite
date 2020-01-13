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
import re
import time
import sys
import copy
import socket

try:
    from . import gpu
except:
    import gpu

try:
    from . import logging
except:
    import logging


def print_debug_log(*items):
    """
        Simple function for handling printing.
        In case we want to change behavior in the future to logging.
        :param items:
        :return:
    """
    print(logging.warning_str("DEBUG::: "), logging.info_str(*items))


def print_error_log(*items):
    """

    :param items:
    :return:
    """
    print(logging.fail_str("ERROR::: "), logging.warning_str(*items))


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


def create_connections(connections, username, password, port=9999, is_debug=True):
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
        if is_debug:
            print_debug_log(f"Connecting to => URL: {url}. Client name: {client_name}")
        try:
            result.append(SshConnection(url, username, password, client_name=client_name, port=port, is_debug=is_debug))
        except AttributeError as e:
            print_error_log(f"Failed to connect to {url}. Please check your credentials.")

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
        """
            Get GPU information from "nvidia-smi"
            and append metadata to the information.
            Some metadata include
            - Machine with largest number of free space
            :return:
        """
        result = {'servers': []}

        # first slot represents the value, the second is the name of the item
        min_memory = [sys.maxsize, '']
        min_free_memory = [sys.maxsize, '']
        min_consumed_memory = [sys.maxsize, '']

        max_memory = [0, '']
        max_free_memory = [0, '']
        max_consumed_memory = [0, '']

        def update_min(min_thus_far, current_value, name):
            if min_thus_far[0] > current_value:
                min_thus_far[0], min_thus_far[1] = current_value, name

        def update_max(max_thus_far, current_value, name):
            if max_thus_far[0] < current_value:
                max_thus_far[0], max_thus_far[1] = current_value, name

        # GPU Order
        gpu_order = []

        # Get gpu info
        for connection in self.connections:
            gpu_info, server_stats = connection.gpu_info()
            server_name = connection.client_name
            server_dict = {'name': server_name, "gpu": gpu_info, "stats": server_stats}

            # For sorting
            deep_copy_gpu = copy.deepcopy(gpu_info)
            for gpu in deep_copy_gpu:
                gpu['name'] = server_name

            # CCalculate statistics
            total_memory, total_free_memory, total_consumed_memory = \
                server_stats['total_memory'], server_stats['total_free_memory'], server_stats['total_consumed_memory']

            # Update min
            update_min(min_memory, total_memory, server_name)
            update_min(min_free_memory, total_free_memory, server_name)
            update_min(min_consumed_memory, total_consumed_memory, server_name)

            # Update max
            update_max(max_memory, total_memory, server_name)
            update_max(max_free_memory, total_free_memory, server_name)
            update_max(max_consumed_memory, total_consumed_memory, server_name)

            # Append results
            result['servers'].append(server_dict)
            gpu_order.extend(deep_copy_gpu)

        # Update statistics
        result['min_memory'] = min_memory
        result['min_free_memory'] = min_free_memory
        result['min_consumed_memory'] = min_consumed_memory

        result['max_memory'] = max_memory
        result['max_free_memory'] = max_free_memory
        result['max_consumed_memory'] = max_consumed_memory
        # Show free gpus
        result['free_gpus'] = sorted(gpu_order, key=lambda i: i['free_memory'], reverse=True)

        return result

    def __getitem__(self, position):
        return self.connections[position]

    def __len__(self):
        return len(self.connections)

    def __repr__(self):
        return f"<class>SshConnectionManager. Clients: {self.connections}"


class SshConnection:
    """
        An SshConnection object represents a connection between a remote machine and the
        local host.
    """

    def __init__(self, url, username, password, client_name, port=22, is_debug=False, use_interactive_shell=True):
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
        self.use_interactive_shell = use_interactive_shell

        # Open
        self.open()

        if use_interactive_shell:
            self.channel = self.ssh.invoke_shell()

    def open(self):
        """
            Access remote or local server via ssh
            :return:
        """
        if self._isalive():
            if self.is_debug:
                print_debug_log("connection is still alive!")
            return
        if self.is_debug:
            print_debug_log(f"Attempting the following command: ssh {self.username}@{self.url} -p {self.port}")
        try:
            self.ssh.connect(self.url, self.port, self.username, self.password)
        except socket.gaierror as e:
            e_str = str(e)
            print_error_log(f"Failed to connect: {e_str}")

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
            if self.is_debug:
                print_debug_log(f"Successfully closed: {self}")

    def cmd(self, shell_command, close_after=False, do_on_finished=None):
        """
            :param shell_command: Command can either be string or an iterable set of commands
            :param close_after: If set to true, close the connection after running commands
            :param do_on_finished:
            :return:
        """
        self.open()
        if self.use_interactive_shell:
            command_handler = self._execute_interactive_cmd
        else:
            command_handler = self._execute_cmd
        # Pass command
        if isiterable(shell_command):
            for command in shell_command:
                output_data, has_errors = command_handler(command)
                output = ''.join(output_data)
                if self.is_debug:
                    print_debug_log(f"Command executed: '{command}'. Output: {output}. has errors: {has_errors}")
        elif type(str):
            output_data, has_errors = command_handler(shell_command)
            output = ''.join(output_data)
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
        errors = stderr.readlines()
        has_errors = False
        if len(errors) > 0:
            has_errors = True
        return outlines, has_errors

    def _execute_interactive_cmd(self, cmd):
        """
                :param cmd: the command to be executed on the remote computer
                :examples:  execute('ls')
                            execute('finger')
                            execute('cd folder_name')
                """
        cmd = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', cmd).replace('\b', '').replace('\r', '')
        self.channel.send(cmd + '\n')
        time.sleep(0.5)
        while True:
            if self.channel.recv_ready():
                channel_data = self.channel.recv(65535).decode("utf-8")
                has_errors = self.channel.recv_stderr_ready()
                return channel_data, has_errors

    def gpu_info(self):
        """
            :return: Returns a list of GPU information via the "nvidia-smi" command
        """
        nvidia_smi_output, has_errors = self._execute_cmd("nvidia-smi")
        if self.is_debug:
            print_debug_log(
                f"------------------------------ {self.client_name} ------------------------------  \n{nvidia_smi_output}")
        return gpu.parse_nvidia_smi_output(nvidia_smi_output)

    def __repr__(self):
        return f"<SshConnection>: [url={self.url}, username={self.username}, client_name={self.client_name}" \
               f", port={self.port}]"

    def __del__(self):
        self.ssh.close()
