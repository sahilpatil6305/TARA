import datetime
import os
import sys
import traceback


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                safe_data = data.encode("ascii", errors="replace").decode("ascii")
                stream.write(safe_data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(logs_dir="logs"):
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"run_{timestamp}.log")
    log_file = open(log_path, "a", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    print(f"[*] Run log file: {log_path}")
    return log_path, log_file, original_stdout, original_stderr


def finalize_run_logging(log_file, original_stdout, original_stderr):
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def log_unhandled_exception(exc):
    print("[!] Unhandled exception during pipeline execution.")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
