"""Summarize transport traces after a drained diagnostic run, without combining process timelines.

Usage: python3 transport_summary.py RUN_DIRECTORY
Writes transport-summary.json beside client-trace.json and stderr.log. Missing or inconsistent traces fail.
"""
import argparse
import json
import math
from pathlib import Path

PREFIX = 'haas-transport '


def require(condition, message):
    if not condition:
        raise ValueError(message)


def point(row, name, scale):
    value = row.get(name)
    require(isinstance(value, list) and len(value) == 2, f'Missing point: {name}')
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in value), f'Invalid point: {name}')
    require(value[0] >= 0 and (value[1] == -1 or value[1] >= 0), f'Unmeasured point: {name}')
    return value[0] * scale, None if value[1] == -1 else value[1] * scale


def span(row, begin, end, scale=1.0):
    left, right = point(row, begin, scale), point(row, end, scale)
    require(right[0] >= left[0], f'Negative wall interval: {begin} -> {end}')
    cpu = None if left[1] is None or right[1] is None else right[1] - left[1]
    require(cpu is None or cpu >= 0, f'Negative thread CPU interval: {begin} -> {end}')
    return {'wall': right[0] - left[0], 'thread_cpu': cpu}


def summed(rows, begin, end, scale=1.0):
    values = [span(row, begin, end, scale) for row in rows]
    return {'wall': sum(value['wall'] for value in values),
            'thread_cpu': None if any(value['thread_cpu'] is None for value in values)
            else sum(value['thread_cpu'] for value in values)}


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, f'Duplicate {key}: {value}')
        result[value] = row
    return result


def ordered(row, names, scale=1.0):
    for left, right in zip(names, names[1:]):
        span(row, left, right, scale)


def summarize(client, events):
    require(client['schema'] == 1, 'Unknown client trace schema.')
    require(client['clock'] == 'perf_counter_seconds/thread_time_seconds', 'Unknown client clock.')
    require(events and all(row['schema'] == 1 for row in events), 'Missing or unknown worker trace schema.')
    require(events[0]['event'] == 'drained', 'Worker trace is not emitted after drain.')
    require(all(row['event'] in {'drained', 'line', 'job'} for row in events), 'Unknown worker event.')
    drained = [row for row in events if row['event'] == 'drained']
    require(len(drained) == 1, 'Expected one drained marker.')
    cpp_lines = unique([row for row in events if row['event'] == 'line'], 'line')
    jobs = unique([row for row in events if row['event'] == 'job'], 'serial')
    sent = unique(client['lines'], 'line')
    received = unique(client['received'], 'response_line')
    requests = unique(client['requests'], 'request')
    require(requests, 'No measured requests.')
    require(len(cpp_lines) == drained[0]['lines'] and len(jobs) == drained[0]['jobs'], 'Incomplete worker trace.')
    require(set(cpp_lines) == set(sent) == set(range(1, len(sent) + 1)), 'Input line coverage differs.')
    require(set(requests) == set(range(1, len(requests) + 1)), 'Request ordinals are not contiguous.')
    require(set(received) == set(range(1, len(received) + 1)), 'Response ordinals are not contiguous.')
    eof = point(drained[0], 'eof', 1e-9)
    for row in cpp_lines.values():
        ordered(row, ['read_begin', 'read_end', 'locked', 'parse_begin', 'parse_end', 'dispatch_end'], 1e-9)
        require(eof[0] >= row['dispatch_end'][0] * 1e-9, 'EOF precedes an input line.')
    for row in received.values():
        ordered(row, ['read_begin', 'read_end', 'parse_begin', 'parse_end', 'enqueue_begin', 'enqueue_end'])
    rows, used_serials, used_lines = [], set(), set()
    for request_index, request in requests.items():
        numbers = list(range(request['first_line'], request['last_line'] + 1))
        require(len(numbers) >= 2 and set(numbers) <= set(sent), 'A request needs chunks and transcribe.')
        require(not used_lines.intersection(numbers), 'Input lines belong to multiple requests.')
        used_lines.update(numbers)
        inputs = [sent[number] for number in numbers]
        cpp = [cpp_lines[number] for number in numbers]
        serial = cpp[-1]['serial']
        require(serial > 0 and serial not in used_serials and serial in jobs, 'Missing or reused request serial.')
        used_serials.add(serial)
        for py, native in zip(inputs, cpp):
            require(py['request'] == request_index and native['serial'] == serial, 'Request/serial mismatch.')
            require(all(py[key] == native[key] for key in ['bytes', 'kind', 'seq']), 'Input line metadata differs.')
            ordered(py, ['json_begin', 'json_end', 'write_begin', 'write_end', 'flush_end'])
        require([row['kind'] for row in inputs] == ['chunk'] * (len(inputs) - 1) + ['transcribe'], 'Unexpected request line kinds.')
        require([row['seq'] for row in inputs[:-1]] == list(range(len(inputs) - 1)), 'Chunk sequence differs.')
        for row in cpp[:-1]:
            ordered(row, ['parse_end', 'decode_begin', 'decode_end', 'append_begin', 'append_end', 'dispatch_end'], 1e-9)
        for row in inputs[:-1]:
            ordered(row, ['base64_begin', 'base64_end', 'json_begin'])
        ordered(cpp[-1], ['parse_end', 'queued', 'dispatch_end'], 1e-9)
        job = jobs[serial]
        require(job['recognition'], 'Only recognition jobs have complete worker boundaries.')
        ordered(job, ['taken', 'entered', 'api_begin', 'api_end', 'terminal_ready', 'finish_returned'], 1e-9)
        require(job['taken'][0] >= cpp[-1]['dispatch_end'][0], 'Job starts before the input lock is released.')
        require(all(math.isfinite(request[key]) for key in ['start', 'end', 'elapsed']), 'Invalid request timer.')
        elapsed = request['end'] - request['start']
        require(elapsed >= 0 and math.isclose(elapsed, request['elapsed'], rel_tol=0, abs_tol=1e-12), 'Request timer differs.')
        require(inputs[0]['base64_begin'][0] >= request['start'] and inputs[-1]['flush_end'][0] <= request['end'],
                'Client send is outside the request timer.')
        response = received[request['response_line']]
        waits = [row for row in client['waits'] if row['response_line'] == request['response_line']]
        require(len(waits) == 1, 'Expected one wait for the terminal response.')
        wait = waits[0]
        require(response['parse_end'][0] <= wait['end'][0] <= request['validate_begin'][0] <= request['end'],
                'Terminal response boundaries differ.')
        # This duration is wholly inside the Python request, but its timestamps have a separate clock origin.
        # The first getline begins before a client may start; finish_returned can follow its timer's end.
        contained = (job['terminal_ready'][0] - cpp[0]['read_end'][0]) * 1e-9
        require(0 <= contained <= elapsed, 'Worker contained span exceeds the paired client request.')
        rows.append({
            'request': request_index, 'serial': serial, 'warmup': request['warmup'], 'client_request_seconds': elapsed,
            'client': {
                'base64_with_slice': summed(inputs[:-1], 'base64_begin', 'base64_end'),
                'json_dumps': summed(inputs, 'json_begin', 'json_end'),
                'textio_write': summed(inputs, 'write_begin', 'write_end'),
                'textio_flush': summed(inputs, 'write_end', 'flush_end'),
                'terminal_stdout_readline_including_idle': span(response, 'read_begin', 'read_end'),
                'terminal_json_loads': span(response, 'parse_begin', 'parse_end'),
                'terminal_queue_get': span(wait, 'begin', 'end'),
                'terminal_validation_wall': request['end'] - request['validate_begin'][0],
            },
            'worker': {
                'first_getline_including_idle': span(cpp[0], 'read_begin', 'read_end', 1e-9),
                'all_getline_including_idle': summed(cpp, 'read_begin', 'read_end', 1e-9),
                'input_lock_wait': summed(cpp, 'read_end', 'locked', 1e-9),
                'parse_json': summed(cpp, 'parse_begin', 'parse_end', 1e-9),
                'base64_decode': summed(cpp[:-1], 'decode_begin', 'decode_end', 1e-9),
                'pcm_append': summed(cpp[:-1], 'append_begin', 'append_end', 1e-9),
                'queue_to_taken_wall': (job['taken'][0] - cpp[-1]['queued'][0]) * 1e-9,
                'request_setup': span(job, 'entered', 'api_begin', 1e-9),
                'transcribe_api': span(job, 'api_begin', 'api_end', 1e-9),
                'terminal_construction': span(job, 'api_end', 'terminal_ready', 1e-9),
                'finish_lock_write_flush': span(job, 'terminal_ready', 'finish_returned', 1e-9),
            },
            'contained_first_read_end_to_terminal_ready_seconds': contained,
            'unattributed_outside_contained_seconds': elapsed - contained,
        })
    require(used_lines == set(sent) and used_serials == set(jobs), 'Unmatched input lines or jobs.')
    return {'schema': 1, 'units': 'seconds', 'syscalls_measured': False, 'rows': rows,
            'interpretation': [
                'Python and C++ clock epochs are independent; only one paired contained duration is subtracted.',
                'Client and worker stages overlap. Do not sum them or nested/overlapping spans.',
                'getline/readline wall time includes producer idle, scheduler delay, and pipe waits.',
                'Thread CPU is available only within one thread; null means unmeasured, not zero.',
                'Wall minus thread CPU does not identify a syscall or prove that stdio synchronization is the cause.',
                'finish_lock_write_flush may extend beyond the client request and is not subtracted from it.',
                'The remaining duration is unattributed. It is not a measured IPC, syscall, or CPU cost.',
            ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    client = json.loads((args.directory / 'client-trace.json').read_text())
    events = [json.loads(line[len(PREFIX):]) for line in (args.directory / 'stderr.log').read_text().splitlines()
              if line.startswith(PREFIX)]
    report = summarize(client, events)
    (args.directory / 'transport-summary.json').write_text(json.dumps(report, indent=2) + '\n')
