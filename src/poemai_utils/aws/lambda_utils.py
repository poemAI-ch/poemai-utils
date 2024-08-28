import json


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
    if "queryStringParameters" in event:
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
        return json.loads(event["body"])

    return params
