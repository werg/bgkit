from bgkit_core.host_memory_guard import watch


def test_guard_aborts_after_consecutive_low_readings_only():
    fired = []
    readings = iter([50.0, 5.0, 5.0, 40.0, 4.0, 3.0, 2.0, 1.0])
    watch(10.0, interval=0.0, grace_checks=3, read=lambda: next(readings),
          abort=lambda a, f: fired.append((a, f)), max_checks=8)
    # two low readings, a recovery (counter resets), then three in a row
    assert fired == [(2.0, 10.0)]


def test_guard_ignores_unreadable_meminfo():
    fired = []
    watch(10.0, interval=0.0, grace_checks=1, read=lambda: None,
          abort=lambda a, f: fired.append(a), max_checks=5)
    assert fired == []
