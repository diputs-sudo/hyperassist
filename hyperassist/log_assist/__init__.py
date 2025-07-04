from .analyzer import process_from_file, process_from_stdin

def process(log_path=None, num_classes=2):
    """
    Main entrypoint for log analysis.
    If log_path is provided, analyzes file. If None, reads from stdin.
    """
    if log_path:
        return process_from_file(log_path, num_classes=num_classes)
    else:
        return process_from_stdin(num_classes=num_classes)

import sys

_buffer = []
_old_stdout = None

def live():
    """Begin capturing important log lines live from stdout."""
    global _buffer, _old_stdout
    class InterceptStdout:
        def write(self, data):
            sys.__stdout__.write(data)
            if any(k in data for k in ("loss", "acc", "grad_norm", "epoch", "nan")):
                _buffer.append(data)
        def flush(self):
            sys.__stdout__.flush()
    _old_stdout = sys.stdout
    sys.stdout = InterceptStdout()

def summarize_live(num_classes=2):
    """Analyze and summarize the buffered important log lines."""
    global _buffer, _old_stdout
    if _old_stdout is not None:
        sys.stdout = _old_stdout
    from .analyzer import parse_log_into_chunks, analyze_chunks, print_chunk_results, analyze_global
    lines = [line for chunk in _buffer for line in chunk.splitlines()]
    chunks, grad_norms = parse_log_into_chunks(lines)
    analyze_chunks(chunks, grad_norms, num_classes=num_classes)
    print_chunk_results(chunks)
    analyze_global(chunks)
    _buffer.clear()

__all__ = ["process", "live", "summarize_live"]
