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