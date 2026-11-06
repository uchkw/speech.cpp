"""Check the thread sweep order without a library, model or worker execution.

Usage: python3 021-thread-sweep-check.py [PLAN_JSON]
"""
import json
from pathlib import Path
import runpy
import sys


sweep = runpy.run_path(str(Path(__file__).with_name('021-thread-sweep.py')))
names = list(sweep['conditions'](16))
orders = [sweep['condition_order'](names, index) for index in range(3)]
for order in orders:
    assert len(order) == 9 and set(order) == set(names), 'Each condition must run exactly once per round.'
for first, last in [('candidate-1', 'candidate-16'), ('candidate-2', 'candidate-8')]:
    assert orders[0].index(first) < orders[0].index(last), 'The first round must use the stated thread-count order.'
    assert orders[1].index(first) > orders[1].index(last), 'The second round must reverse thread-count order.'
assert orders[2][0] != orders[0][0], 'The third round must move the starting condition.'
assert names == list(sweep['conditions'](16)), 'Ordering must not mutate the condition definitions.'
if len(sys.argv) == 2:
    plan = json.loads(Path(sys.argv[1]).read_text())
    expected = []
    for kind in plan['types']:
        for index in range(plan['rounds']):
            expected.append({'type': kind, 'round': index,
                             'conditions': sweep['condition_order'](plan['conditions'], index)})
    assert plan['execution_order'] == expected, 'The plan must save its actual condition order.'
    jobs = [json.loads(Path(job['spec']).read_text()) for job in plan['jobs']]
    actual = [(job['model']['type'], job['round'], job['side']) for job in jobs]
    wanted = [(row['type'], row['round'], name) for row in expected for name in row['conditions']]
    assert actual == wanted, 'The worker jobs must follow the recorded order.'
elif len(sys.argv) != 1:
    raise SystemExit('usage: 021-thread-sweep-check.py [PLAN_JSON]')
print('Thread-sweep order: passed; no worker, library, model or GPU was invoked.')
