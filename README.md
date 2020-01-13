# Excite

## What is Excite

A tool for researchers to effectively run and keep track of experiments on remote servers.

## Features

Below is a list of features.
- Extracting GPU info of nvidia-based GPUs
- Running scripts and shell commands remotely 
- Managing multiple ssh sessions
- Create and manage tmux sessions remotely
- Built-in debugging features

## Dependencies

Requires: Python 3.5 >= (not tested on lower versions, but may still work).
List of dependent libraries
- [Paramiko](https://github.com/paramiko/paramiko) - For interacting with remote servers via ssh.

## Demos 

A very simple demo can be found in the demo folder. 

### Creating sessions

Built on top of paramiko, it creates ssh sessions that allow scripts and commands to be run 
programatically on a remote server.

```python

port = 9999
username = 'username'
password = 'password'

servers = [f"server{i}.domain.co.kr" for i in range(2, 8)]
tmux_session_names = [f"tmux_session_names{i}" for i in range(2, 8)]

# Create the ssh connection objects.
# Each connection represents an ssh connection to the user
connections = ssh.create_connections(servers, username, password, port=port, is_debug=True)
# Create sessions
manager = ssh.SshConnectionManager(connections)

```

### GPU Information

excite can print the gpu usage information, along with statistical information such as available number of free memory, as well as a list of gpus
with the largest amount of free memory.

```python

port = 9999
username = 'username'
password = 'password'

servers = [f"server{i}.domain.co.kr" for i in range(2, 8)]
tmux_session_names = [f"tmux_session_names{i}" for i in range(2, 8)]

# Create the ssh connection objects.
# Each connection represents an ssh connection to the user
connections = ssh.create_connections(servers, username, password, port=port, is_debug=True)
# Create sess
manager = ssh.SshConnectionManager(connections)

# Print GPU info:
print_gpu_info(manager)

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


```

### Creating Tmux sessions and executing commands remotely

```python
servers = [f"server{i}.domain.co.kr" for i in range(1, 8)]
tmux_session_names = [f"tmux_session_names{i}" for i in range(1, 8)]
# Create the ssh connection objects.
# Each connection represents an ssh connection to the user
connections = ssh.create_connections(servers, username, password, port=port, is_debug=True)
# Create sess
manager = ssh.SshConnectionManager(connections)

# For all the servers enlisted,
# perform the following simple commands
for i, connection in enumerate(manager):
    # Print connection info
    print(connection)
    # Perform the following commands
    connection.cmd(
        [
            # 1. Create tmux session with name "tmux_session_names_{i}" where i is a number. e.g. 1
            # "tmux new-session -s tmux_session_names_{i]"
            Tmux.new_session(tmux_session_names[i]),
            # 2. "tmux ls"
            Tmux.ls(),
            # 3. ctrl + b, d
            Tmux.detach()
        ])
```