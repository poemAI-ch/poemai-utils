import os
import threading
import unittest.mock

import pytest
from poemai_utils.operations_utils import GracefulStopper


class TestGracefulStopperEnhanced:
    def test_graceful_stopper_in_main_thread_normal_environment(self):
        """Test that GracefulStopper works normally in main thread without serverless environment."""
        # Ensure we're not in a serverless environment
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            # Remove any Lambda environment variables that might be present
            env_vars_to_remove = [
                "AWS_LAMBDA_FUNCTION_NAME",
                "LAMBDA_TASK_ROOT",
                "_HANDLER",
                "AWS_EXECUTION_ENV",
            ]
            for var in env_vars_to_remove:
                if var in os.environ:
                    del os.environ[var]

            # Mock threading.current_thread to return main thread
            with unittest.mock.patch("threading.current_thread") as mock_thread:
                mock_thread.return_value = threading.main_thread()

                stopper = GracefulStopper()

                # In normal environment with main thread, signal handling should be enabled
                assert stopper.signal_handling_enabled
                assert not stopper.stop_now
                assert stopper.stop_now_requests == 0

    def test_graceful_stopper_in_lambda_environment(self):
        """Test that GracefulStopper gracefully handles Lambda environment."""
        # Mock Lambda environment
        with unittest.mock.patch.dict(
            os.environ,
            {
                "AWS_LAMBDA_FUNCTION_NAME": "test-function",
                "LAMBDA_TASK_ROOT": "/var/task",
            },
        ):
            # Mock threading.current_thread to return main thread (still main thread in Lambda)
            with unittest.mock.patch("threading.current_thread") as mock_thread:
                mock_thread.return_value = threading.main_thread()

                stopper = GracefulStopper()

                # In Lambda environment, signal handling should be disabled
                assert not stopper.signal_handling_enabled
                assert not stopper.stop_now
                assert stopper.stop_now_requests == 0

    def test_graceful_stopper_in_non_main_thread(self):
        """Test that GracefulStopper gracefully handles non-main thread."""
        # Create a mock thread that's not the main thread
        mock_thread = threading.Thread()

        with unittest.mock.patch("threading.current_thread") as mock_current_thread:
            mock_current_thread.return_value = mock_thread

            stopper = GracefulStopper()

            # In non-main thread, signal handling should be disabled
            assert not stopper.signal_handling_enabled
            assert not stopper.stop_now
            assert stopper.stop_now_requests == 0

    def test_graceful_stopper_with_aws_execution_env(self):
        """Test that GracefulStopper detects AWS_EXECUTION_ENV as serverless."""
        with unittest.mock.patch.dict(
            os.environ, {"AWS_EXECUTION_ENV": "AWS_Lambda_python3.12"}
        ):
            with unittest.mock.patch("threading.current_thread") as mock_thread:
                mock_thread.return_value = threading.main_thread()

                stopper = GracefulStopper()

                # AWS_EXECUTION_ENV should trigger serverless detection
                assert not stopper.signal_handling_enabled
                assert not stopper.stop_now
                assert stopper.stop_now_requests == 0

    def test_graceful_stopper_signal_handling_exception(self):
        """Test that GracefulStopper handles signal.signal exceptions gracefully."""
        # Mock signal.signal to raise ValueError (simulating the original error)
        with unittest.mock.patch(
            "poemai_utils.operations_utils.signal.signal"
        ) as mock_signal:
            mock_signal.side_effect = ValueError(
                "signal only works in main thread of the main interpreter"
            )

            with unittest.mock.patch("threading.current_thread") as mock_thread:
                mock_thread.return_value = threading.main_thread()

                # Clear environment variables to simulate normal environment
                with unittest.mock.patch.dict(os.environ, {}, clear=False):
                    env_vars_to_remove = [
                        "AWS_LAMBDA_FUNCTION_NAME",
                        "LAMBDA_TASK_ROOT",
                        "_HANDLER",
                        "AWS_EXECUTION_ENV",
                    ]
                    for var in env_vars_to_remove:
                        if var in os.environ:
                            del os.environ[var]

                    stopper = GracefulStopper()

                    # Even with signal.signal raising exception, stopper should work
                    assert not stopper.signal_handling_enabled
                    assert not stopper.stop_now
                    assert stopper.stop_now_requests == 0

    def test_graceful_stopper_exit_gracefully_method(self):
        """Test that the exit_gracefully method works correctly."""
        # Create stopper without signal handling (to avoid actual signal setup)
        with unittest.mock.patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "test"}):
            stopper = GracefulStopper()

            # Simulate receiving a signal
            stopper.exit_gracefully()

            assert stopper.stop_now
            assert stopper.stop_now_requests == 1

            # Second signal
            stopper.exit_gracefully()
            assert stopper.stop_now_requests == 2

            # Third signal should trigger exit (but we'll mock it)
            with unittest.mock.patch("builtins.exit") as mock_exit:
                stopper.exit_gracefully()
                assert stopper.stop_now_requests == 3
                mock_exit.assert_called_once_with(1)
