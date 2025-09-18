from poemai_utils.state_machine import StateMachine


def test_state_machine_transitions():
    transitions = {"idle": {"start": "running"}, "running": {"stop": "idle"}}
    sm = StateMachine(initial_state="idle", transitions=transitions)
    sm.process_event("start")
    assert sm.current_state == "running"
    sm.process_event("stop")
    assert sm.current_state == "idle"


def test_state_machine_implicit_self_transitions():
    transitions = {"idle": {"start": "running"}, "running": {"stop": "idle"}}
    sm = StateMachine(
        initial_state="idle", transitions=transitions, implicit_self_transitions=True
    )
    sm.process_event("pause")  # Not defined, should stay in 'idle'
    assert sm.current_state == "idle"


def test_state_machine_error_state():
    transitions = {"idle": {"start": "running"}, "running": {"stop": "idle"}}
    sm = StateMachine(
        initial_state="idle", transitions=transitions, error_state="error"
    )
    sm.process_event("unknown")  # Not defined, should transition to 'error'
    assert sm.current_state == "error"
