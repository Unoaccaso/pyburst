"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Friday, January 26th 2024, 4:53:21 pm
Author: Riccardo Felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import platform


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.folders_to_watch = ["../", "../../qp_transform/utils"]

    def on_any_event(self, event):
        print(f"Received event: {event.event_type}, {event.src_path}")
        if event.is_directory:
            return
        if event.src_path.endswith(".py"):
            restart_server()


def restart_server():
    print("Restarting Bokeh Server...")
    if server_process:
        server_process.terminate()
        server_process.wait()
    start_server()


def start_server():
    global server_process
    args = ["python", os.path.join(PATH_TO_THIS, "gw_inspector.py"), "--use-xheaders"]

    if platform.system() == "Windows":
        # Utilizza STARTUPINFO per avviare il processo senza una nuova finestra
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        server_process = subprocess.Popen(args, startupinfo=startupinfo)
    else:
        # Utilizza un subprocesso senza shell su sistemi Unix-like
        server_process = subprocess.Popen(args, preexec_fn=os.setsid)


if __name__ == "__main__":
    server_process = None

    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()

    try:
        start_server()
        observer.join()
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
