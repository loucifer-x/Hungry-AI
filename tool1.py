import subprocess
import sys
import threading
import itertools
import time

def run_command(command: list):
    result = subprocess.run(
        command,
        text=True,
        capture_output=True
    )
    return result.stdout, result.stderr, result.returncode


def spinner(stop_event):
    for frame in itertools.cycle(["thinking.", "thinking..", "thinking..."]):
        if stop_event.is_set():
            break
        sys.stdout.write("\r" + frame)
        sys.stdout.flush()
        time.sleep(0.3)
    sys.stdout.write("\r")


def tool1():
    linuxpri = [
        ["id"],
        ["sudo", "-l"],
        ["find", "/", "-type", "f", "-perm", "-04000", "-ls"],
        ["cat", "/etc/crontab"],
        ["ls", "-la", "/etc/cron.*"],
        ["find", "/", "-writable", "-type", "d"],
        ["cat", "/etc/passwd"]
    ]

    full_output = ""

    for cmd in linuxpri:
        stop_event = threading.Event()
        t = threading.Thread(target=spinner, args=(stop_event,))
        t.start()

        stdout, stderr, code = run_command(cmd)

        stop_event.set()
        t.join()

        print(f"\n>>> {cmd}")

        full_output += f"\n>>> {cmd}\n"
        full_output += stdout

        if stderr:
            full_output += "\n[stderr]\n" + stderr

        full_output += "\n" + "-" * 50 + "\n"
    print(full_output)
    return "Find the a possible linux privilege escalation" + ", ".join(full_output)
        