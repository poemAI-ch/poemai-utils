import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

_logger = logging.getLogger(__name__)

"""
    A lightweight state machine for modeling event-driven code in a structured way.

    This class replaces complex if-elif-else ladders by defining states and
    transitions between them, providing a clearer and more maintainable system.

    Attributes:
        initial_state (str): The starting state of the state machine.
        transitions (Dict[str, Dict[str, str]]): Describes how states transition on given events.
        error_state (Optional[str]): Default state if an event does not have a defined transition.
        implicit_self_transitions (bool): If True, self-transitions are allowed if no other
            transition is specified.
        log_transitions (bool): Whether to log the state transitions.
        current_state (str): Holds the current state of the state machine.

    Example:
        Consider a state machine with three states ('idle', 'waiting_for_answer', 'processing_answer')
        and events ('trigger_send', 'received', 'processing_finished').

        states = {
            'idle': { 'trigger_send': 'waiting_for_answer' },
            'waiting_for_answer': { 'received': 'processing_answer' },
            'processing_answer': { 'processing_finished': 'idle' }
        }

        sm = StateMachine(initial_state='idle', transitions=states)

        sm.process_event('trigger_send')
        assert sm.current_state == 'waiting_for_answer'

        sm.process_event('received')
        assert sm.current_state == 'processing_answer'

        sm.process_event('processing_finished')
        assert sm.current_state == 'idle'
    """


@dataclass
class StateMachine:
    initial_state: str
    transitions: Dict[str, Dict[str, str]]
    error_state: Optional[str] = None
    implicit_self_transitions: bool = False
    log_transitions: bool = False
    current_state: str = field(init=False)

    def __post_init__(self):
        self.current_state = self.initial_state

    def process_event(self, event: Optional[str]):
        if event is None:  # Non-event, no transition
            return

        # Retrieve possible transitions from the current state
        next_state = self.transitions.get(self.current_state, {}).get(event)

        # Log transition if requested
        if self.log_transitions:
            if next_state:
                _logger.info(
                    f"Transitioning from {self.current_state} to {next_state} with event '{event}'"
                )
            elif self.error_state:
                _logger.info(
                    f"Transitioning from {self.current_state} to error state '{self.error_state}' due to event '{event}' -  no explicit transition defined"
                )
            elif self.implicit_self_transitions:
                _logger.info(
                    f"Implicit self-transition in {self.current_state} with event '{event}'"
                )
            else:
                _logger.info(
                    f"Unspecified transition from {self.current_state} with event '{event}'"
                )

        # Update state based on transition
        if next_state is not None:
            self.current_state = next_state
        elif self.error_state:
            self.current_state = self.error_state
        elif self.implicit_self_transitions:
            # Stay in the current state if implicit self-transitions are allowed
            pass
        else:
            raise ValueError(
                f"Unspecified transition from state '{self.current_state}' with event '{event}'"
            )
