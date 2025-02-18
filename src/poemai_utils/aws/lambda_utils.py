import json
import logging
import random
import time

_logger = logging.getLogger(__name__)


def extract_parameters(event, context):
    """
    Extracts parameters from a Lambda function event triggered by either API Gateway or SQS.
    This utility function determines the source of the event and extracts parameters accordingly,
    handling both API Gateway requests (with query string and path parameters) and SQS messages.

    Args:
        event (dict): The event dictionary passed to the Lambda handler.
        context (LambdaContext): The runtime context of the Lambda function.

    Returns:
        dict or list: If the event is from SQS and contains multiple records, returns a list of dictionaries
                      with each containing parameters from one SQS message. If from API Gateway or a single
                      SQS message, returns a single dictionary of parameters.
    """

    def json_or_string(body):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    # Check if this is an SQS message
    if "Records" in event and event["Records"][0].get("eventSource") == "aws:sqs":
        messages = [json_or_string(record["body"]) for record in event["Records"]]
        return messages if len(messages) > 1 else messages[0]

    # API Gateway request: Combine query string and path parameters
    params = {}
    if "queryStringParameters" in event and event["queryStringParameters"] is not None:
        params.update(event["queryStringParameters"])
    if "pathParameters" in event and event["pathParameters"] is not None:
        params.update(event["pathParameters"])

    found_http_data = False
    for k in ["httpMethod", "headers", "resource", "path", "requestContext"]:
        if k in event:
            found_http_data = True
            params[k] = event[k]
    if found_http_data and "body" in event:
        params["body"] = event["body"]

    if "source" in event and event["source"] == "aws.events":
        # EventBridge event, just merge in the event data
        params.update(event)

    # Handle direct invoke or test event with a JSON body
    if not params and "body" in event:
        try:
            return json.loads(event["body"])
        except json.JSONDecodeError:
            return event["body"]

    # Return the original event if no parameters were found (direct invocation fallback)
    return params if params else event


def invoke_lambda_with_backoff(
    lambda_client,
    function_name,
    payload,
    max_attempts=8,
    initial_delay=1,
    invocation_type="RequestResponse",
):
    _logger.info(f"Got lambda_client: {lambda_client}")
    for attempt in range(max_attempts):
        try:
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(payload),
            )
            # Read the payload from the response
            response_payload = (
                response["Payload"].read().decode("utf-8")
            )  # Decoding from bytes to string
            try:
                response_payload = json.loads(
                    response_payload
                )  # Convert string to JSON if possible
            except json.JSONDecodeError:
                pass  # Keep as string if it's not JSON

            response["Payload"] = (
                response_payload  # Replace the StreamingBody with the actual content
            )
            return response
        except lambda_client.exceptions.TooManyRequestsException:
            if attempt < max_attempts - 1:
                sleep_time = initial_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
                _logger.debug(
                    f"Rate exceeded, retrying after {sleep_time:.2f} seconds..."
                )
            else:
                _logger.error(
                    "Max retry attempts reached, unable to invoke lambda function."
                )
                raise  # Re-raise the exception after the last attempt
        except Exception as e:
            _logger.error(f"An error occurred: {str(e)}", exc_info=True)
            raise
