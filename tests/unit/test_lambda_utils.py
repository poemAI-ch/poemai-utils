import json

from poemai_utils.aws.lambda_utils import extract_parameters


def test_extract_parameters():

    body_data = {"name": "Alice", "age": 30}
    bob_data = {"name": "Bob", "age": 25}
    id_data = {"id": "123"}
    full_data = {**body_data, **id_data}

    # Test API Gateway request with query string and path parameters
    event = {
        "queryStringParameters": body_data,
        "pathParameters": id_data,
    }
    assert extract_parameters(event, None) == full_data

    # Test API Gateway request with only query string parameters
    event = {"queryStringParameters": body_data}
    assert extract_parameters(event, None) == body_data

    # Test API Gateway request with only path parameters
    event = {"pathParameters": {"id": "123"}}
    assert extract_parameters(event, None) == {"id": "123"}

    # Test API Gateway request with JSON body
    event = {"body": json.dumps(body_data)}
    assert extract_parameters(event, None) == body_data

    # Test SQS message with single record
    event = {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": json.dumps(body_data),
            }
        ]
    }
    assert extract_parameters(event, None) == body_data

    # Test SQS message with multiple records
    event = {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": json.dumps(body_data),
            },
            {
                "eventSource": "aws:sqs",
                "body": json.dumps(bob_data),
            },
        ]
    }
    assert extract_parameters(event, None) == [body_data, bob_data]

    # test SQS message with only a string, not json
    event = {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": "hello world",
            }
        ]
    }
    assert extract_parameters(event, None) == "hello world"


def test_event_bridge_event():
    event = {
        "version": "0",
        "id": "89d1a02d-5ec7-412e-82f5-13505f849b41",
        "detail-type": "Scheduled Event",
        "source": "aws.events",
        "account": "123456789012",
        "time": "2020-07-28T18:03:33Z",
        "region": "us-west-2",
        "resources": ["arn:aws:events:us-west-2:123456789012:rule/MyRule"],
        "detail": {},
    }

    parameters = extract_parameters(event, None)

    assert parameters["id"] == "89d1a02d-5ec7-412e-82f5-13505f849b41"
    assert parameters["detail-type"] == "Scheduled Event"
    assert parameters["source"] == "aws.events"
    assert parameters["resources"] == [
        "arn:aws:events:us-west-2:123456789012:rule/MyRule"
    ]
