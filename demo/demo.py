from excite import ssh
from excite.cmd import Tmux


def common_commands(session_name, tasks=[]):
    """
        Set of common commands when creating new session
        :param session_name:
        :param tasks:
        :return:
    """
    cmds = [
        Tmux.kill_session(session_name),
        Tmux.new_session(session_name),
        Tmux.attach(session_name),
        "ls -l",
    ]
    cmds.extend(tasks)
    cmds.append(Tmux.detach())
    return cmds


def print_gpu_info(server_manager):
    """
        :param server_manager: Manager object containing a list
        of connections.
        :return:
    """
    gpu_info = server_manager.gpu_info()
    space_count = 6

    # Print stats
    for server in gpu_info['servers']:
        print(server['name'])
        print(server['stats'])
        for gpu in server['gpu']:
            for key in gpu:
                print(f"{key}:\t {gpu[key]}.")
        print("-" * 30)

    # Print manager stats
    max_key = -1

    del gpu_info['servers']
    for stats_key in gpu_info.keys():
        max_key = max(max_key, len(stats_key))
        if stats_key == 'free_gpus':
            print(f"Free GPU: ")
            for item in gpu_info[stats_key][:20]:
                print(f"Server name: {item['name']}. GPU Index: {item['index']}. Free memory: {item['free_memory']}")
        else:
            print(f"{stats_key}:", " " * (space_count + max_key - len(stats_key)), f"{gpu_info[stats_key]}")


if __name__ == "__main__":
    port = 9999
    username = 'username'
    password = 'password'

    # Server and session info
    servers = [f"server{i}.domain.co.kr" for i in range(5, 8)]
    tmux_session_names = [f"tmux_session_names{i}" for i in range(5, 8)]

    # Create the ssh connection objects.
    # Each connection represents an ssh connection to a server
    connections = ssh.create_connections(servers, username, password, port=port, is_debug=True)
    # Create sess
    manager = ssh.SshConnectionManager(connections)

    # Print GPU info:
    print_gpu_info(manager)

    # For all the servers enlisted,
    # perform the simple commands
    for i, connection in enumerate(manager):
        # Print connection info
        print(connection)
        # Perform the following commands
        connection.cmd(
            [
                Tmux.new_session(tmux_session_names[i]),
                Tmux.ls(),
                Tmux.detach()
            ])
