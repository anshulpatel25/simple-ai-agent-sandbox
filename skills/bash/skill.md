# Bash Skill

## Overview
This skill allows the agent to execute bash shell commands inside a sandboxed
Ubuntu Docker container. Every command runs in the same container for the
duration of a session, so state (working directory, environment variables,
installed packages) is preserved across commands.

## Available Commands

### File System
| Command | Purpose | Example |
|---------|---------|---------|
| `ls [path]` | List directory contents | `ls -la /tmp` |
| `pwd` | Print current working directory | `pwd` |
| `cd <path>` | Change directory (use with `&&` chaining) | `cd /tmp && ls` |
| `mkdir <dir>` | Create directory | `mkdir -p /app/data` |
| `rm <path>` | Remove file or directory | `rm -rf /tmp/scratch` |
| `cp <src> <dst>` | Copy file or directory | `cp file.txt /backup/` |
| `mv <src> <dst>` | Move or rename | `mv old.txt new.txt` |
| `touch <file>` | Create empty file | `touch notes.txt` |
| `find <dir> <expr>` | Search for files | `find / -name "*.log"` |
| `tree [path]` | Display directory tree | `tree /app` |

### File Content
| Command | Purpose | Example |
|---------|---------|---------|
| `cat <file>` | Print file contents | `cat /etc/hosts` |
| `head -n <N> <file>` | Print first N lines | `head -n 20 log.txt` |
| `tail -n <N> <file>` | Print last N lines | `tail -n 50 app.log` |
| `grep <pattern> <file>` | Search text in file | `grep -r "error" /var/log` |
| `wc -l <file>` | Count lines | `wc -l data.csv` |
| `echo "text" > file` | Write text to file | `echo "hello" > out.txt` |
| `echo "text" >> file` | Append text to file | `echo "more" >> out.txt` |

### Process & System
| Command | Purpose | Example |
|---------|---------|---------|
| `ps aux` | List running processes | `ps aux` |
| `kill <pid>` | Terminate a process | `kill 1234` |
| `top -bn1` | Snapshot system resource usage | `top -bn1` |
| `df -h` | Disk usage | `df -h` |
| `free -h` | Memory usage | `free -h` |
| `uname -a` | Kernel and system info | `uname -a` |
| `whoami` | Current user | `whoami` |
| `hostname` | Container hostname | `hostname` |
| `env` | Print environment variables | `env` |
| `export VAR=val` | Set environment variable | `export PATH=$PATH:/opt/bin` |

### Networking
| Command | Purpose | Example |
|---------|---------|---------|
| `curl <url>` | HTTP request | `curl https://example.com` |
| `wget <url>` | Download file | `wget https://example.com/file.zip` |
| `ping -c 4 <host>` | Test connectivity | `ping -c 4 google.com` |
| `ip addr` | Show IP addresses | `ip addr` |
| `netstat -tulpn` | Show open ports | `netstat -tulpn` |

### Package Management (APT)
| Command | Purpose | Example |
|---------|---------|---------|
| `apt-get update` | Refresh package index | `apt-get update` |
| `apt-get install -y <pkg>` | Install package | `apt-get install -y curl` |
| `apt-get remove <pkg>` | Uninstall package | `apt-get remove curl` |
| `dpkg -l` | List installed packages | `dpkg -l` |

### Text Processing
| Command | Purpose | Example |
|---------|---------|---------|
| `sed 's/old/new/g' file` | Replace text | `sed 's/foo/bar/g' file.txt` |
| `awk '{print $1}' file` | Extract columns | `awk '{print $2}' data.txt` |
| `sort <file>` | Sort lines | `sort -n numbers.txt` |
| `uniq <file>` | Remove duplicate lines | `sort file.txt \| uniq` |
| `cut -d: -f1 file` | Cut field from lines | `cut -d: -f1 /etc/passwd` |

### Chaining & Control
| Syntax | Purpose | Example |
|--------|---------|---------|
| `cmd1 && cmd2` | Run cmd2 only if cmd1 succeeds | `mkdir /app && cd /app` |
| `cmd1 \| cmd2` | Pipe output of cmd1 to cmd2 | `ps aux \| grep python` |
| `cmd1 ; cmd2` | Run cmd2 regardless of cmd1 result | `ls ; pwd` |
| `cmd > file` | Redirect stdout to file | `ls > files.txt` |
| `cmd 2>&1` | Redirect stderr to stdout | `make 2>&1 \| tail -20` |

## Usage Guidelines
1. **Prefer non-interactive commands** – avoid commands that require user input (e.g., use `apt-get install -y`, not `apt-get install`).
2. **Chain dependent commands** – since `cd` only affects the subprocess, use `cd /path && <command>` to operate in a specific directory.
3. **Check return codes** – a non-zero exit code means the command failed; inspect stderr output.
4. **Be mindful of long-running commands** – commands that run indefinitely will time out.
