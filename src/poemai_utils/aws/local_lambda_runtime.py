import argparse
import logging
import os
import sys

_logger = logging.getLogger(__name__)

"""
This module provides a utility to locally run and emulate AWS Lambda functions inside Docker containers.
It leverages 'awslambdarie' to support running multiple functions under their designated names by dispatching
requests to appropriate host/port configured via environment variables.
It solves the naming constraint by allowing each lambda emulation to operate with its function name via URL parameter,
while internally dispatching to the same function name accepted under the context of 'awslambdarie'.
This makes it possible to locally offer multiple locally emulated lambda functions, and use the boto3 client 'invoke' functionality to call the local lambda functions.

Functions:
    app_setup(): Sets up the FastAPI application, configures the routing to handle function invocations.

Usage:
    Run this module directly to start a local server, which listens for incoming lambda function invocations,
    configured through command-line arguments such as host and port.

Example:
    python local_lambda_runtime.pyExample of Configuring Multiple Lambda Functions:
    To configure multiple AWS Lambda functions, you can set environment variables for each function, mapping "Named Function URLs" to specific endpoints managed by the awslambdarie containers.

    export LAMBDA_FUNCTION_INVOKE_URL_FUNCTION_NAME="http://localhost:PORT"

Replace FUNCTION_NAME with the specific Lambda function's name and PORT with the mapped awslambdarie container's port. For example:
    export LAMBDA_FUNCTION_INVOKE_URL_ASSISTANT="http://localhost:26978"
    export LAMBDA_FUNCTION_INVOKE_URL_MANAGER="http://localhost:26979"

In this configuration, the invoke URL pattern will automatically route requests to the appropriate awslambdarie container based on the function name specified in the path.

For the 'assistant' function, the call:
    http://localhost:26763/2015-03-31/functions/assistant/invocations
will map internally to its endpoint like:
    http://localhost:26978/2015-03-31/functions/function/invocations
assuming the awslambdarie container for "assistant" was started on port 26978.

Similarly, for the 'manager' function with its environment variable set, the invoke URL:
    http://localhost:26763/2015-03-31/functions/manager/invocations
maps to:
    http://localhost:26979/2015-03-31/functions/function/invocations
assuming the awslambdarie container for "manager" was started on port 26979.

To use boto3 to invoke the local lambda functions, you can use the following code snippet:

    import boto3

    client = boto3.client("lambda", endpoint_url="http://localhost:26763")
    response = client.invoke(
        FunctionName="assistant",
        InvocationType="RequestResponse",
        Payload=b'{"key": "value"}',
    )

"""


def app_setup():
    import json

    import requests  # Import necessary modules for handling requests and JSON data
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI()

    def get_function_base_url(function_name: str) -> str:
        # Convert function name to uppercase and replace dashes with underscores
        env_var_name = (
            f"LAMBDA_FUNCTION_INVOKE_URL_{function_name.replace('-', '_').upper()}"
        )
        # Retrieve the base URL from environment variable
        base_url = os.getenv(env_var_name)
        if not base_url:
            raise HTTPException(
                status_code=404, detail=f"Function {function_name} not configured."
            )
        return base_url

    @app.post("/2015-03-31/functions/{function_name}/invocations")
    async def invoke_lambda(function_name: str, request: Request):
        # Extract payload from the request
        payload = await request.json()
        # Get the base URL for the specified function
        base_url = get_function_base_url(function_name)
        if base_url is None:
            raise HTTPException(
                status_code=404, detail=f"Function {function_name} not found."
            )
        # Build the full URL for invocation
        invocation_url = f"{base_url}/2015-03-31/functions/function/invocations"
        _logger.info(f"Forwarding request to {invocation_url}")
        # Forward the request to the actual Lambda function endpoint
        response = requests.post(invocation_url, json=payload)
        return response.json()

    return app


if __name__ == "__main__":
    app = app_setup()

    from uvicorn.main import run

    LOG_FORMAT = (
        "%(asctime)s %(levelname)8s %(name)25s  %(filename)25s:%(lineno)-4d %(message)s"
    )

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Run the local Lambda runtime.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="The host to listen on. Default is 0.0.0.0.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=26763,
        help="The port to listen on. Default is 26763.",
    )

    args = parser.parse_args()

    sys.exit(run(app, host=args.host, port=args.port))
