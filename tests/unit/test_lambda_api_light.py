import json
import logging

from build.lib.poemai_utils.aws.lambda_api_light import JSONResponse
from poemai_utils.aws.lambda_api_light import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    LambdaApiLight,
    Query,
)
from pydantic import BaseModel


class ThingsData(BaseModel):
    thing_key: str
    thing_data: dict


_logger = logging.getLogger(__name__)

_application_service = None


class Application:
    def get_available_thing_keys(self):
        return ["test1", "test2", "test3"]


# Dependency provider function
def get_application_service():
    global _application_service
    if _application_service is None:
        _application_service = Application()

    return _application_service


app = LambdaApiLight()


router = APIRouter()


@app.on_event("startup")
async def startup_event():
    route_info = []
    for route in app.routes:
        route_methods = ",".join(route.methods)
        route_info.append(f"Path: {route.path}, Methods: {route_methods}")
    route_info = sorted(route_info)
    _logger.info("\n ----- Available routes:\n" + "\n".join(route_info) + "\n -----")


PREFIX = f"/test_api/api/v1"
router = APIRouter(prefix=PREFIX)
root_router = APIRouter(prefix=PREFIX)


@root_router.get("/")
async def get_root(
    application_service: Application = Depends(get_application_service),
    x_user_id: str = Header(None),
    x_required_header: str = Header(...),
    the_query: str = Query(None),
):
    return {
        "message": f"Welcome to the api",
        "available_thing_keys": application_service.get_available_thing_keys(),
        "user_id": x_user_id,
        "query": the_query,
        "required_header": x_required_header,
    }


@root_router.get("/query_defaults")
async def get_root(
    application_service: Application = Depends(get_application_service),
    optinal_query_with_default: str = Query("default_value"),
    optional_query_with_none_default: str = Query(None),
    required_query: str = Query(...),
):
    return {
        "optinal_query_with_default": optinal_query_with_default,
        "optional_query_with_none_default": optional_query_with_none_default,
        "required_query": required_query,
    }


@root_router.post("/things/{thing_key}")
async def post_thing(thing_key: str, thing_data: ThingsData):

    headers = {"Location": f"/{thing_key}"}
    return JSONResponse(
        content=thing_data.model_dump(mode="json"),
        status_code=201,
        headers=headers,
    )


@root_router.get("/error")
async def get_error(desired_status_code: int = Query(400)):
    raise HTTPException(status_code=desired_status_code, detail="error")


@root_router.get("/things")
async def get_things():
    return [
        ThingsData(thing_key="one_thing", thing_data={"key": "value"}),
        ThingsData(thing_key="another_thing", thing_data={"key": "value"}),
    ]


@root_router.get("/things/{thing_key}")
async def get_thing(thing_key: str) -> ThingsData:
    return ThingsData(thing_key=thing_key, thing_data={"key": "value"})


app.include_router(root_router)


def test_handle():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "thing_key": "test_thing",
            "the_query": "test_query",
        },
        "headers": {
            "X-User-Id": "test_user_id",
            "X-Required-Header": "required_test_header",
        },
        "body": None,
    }

    response = app.handle_request(event, None)
    _logger.info(f"Response: {response}")

    assert response["statusCode"] == 200

    body_text = response["body"]

    body_obj = json.loads(body_text)

    assert body_obj["message"] == "Welcome to the api"
    assert body_obj["available_thing_keys"] == ["test1", "test2", "test3"]
    assert body_obj["user_id"] == "test_user_id"
    assert body_obj["query"] == "test_query"
    assert body_obj["required_header"] == "required_test_header"

    without_optional_header = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "thing_key": "test_thing",
            "the_query": "test_query",
        },
        "headers": {"X-Required-Header": "required_test_header"},
        "body": None,
    }

    response = app.handle_request(without_optional_header, None)
    assert response["statusCode"] == 200  # headers are optional

    without_optional_header_but_lambda_header_writing = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "thing_key": "test_thing",
            "the_query": "test_query",
        },
        "headers": {"x-required-header": "required_test_header"},
        "body": None,
    }

    response = app.handle_request(without_optional_header, None)
    assert response["statusCode"] == 200  # headers are optional

    without_required_header = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "thing_key": "test_thing",
            "the_query": "test_query",
        },
        "headers": {},
        "body": None,
    }

    response = app.handle_request(without_required_header, None)
    assert response["statusCode"] == 400


def test_query_defaults():

    event_with_no_queries = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/query_defaults",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event_with_no_queries, None)
    assert response["statusCode"] == 400

    event_with_only_required_query = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/query_defaults",
        "queryStringParameters": {
            "required_query": "required_value",
        },
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event_with_only_required_query, None)
    assert response["statusCode"] == 200

    resultbody = json.loads(response["body"])

    assert resultbody["required_query"] == "required_value"
    assert resultbody["optinal_query_with_default"] == "default_value"
    assert resultbody["optional_query_with_none_default"] == None


def test_routes():

    routes = sorted([(r.path, r.methods) for r in app.routes])
    _logger.info(f"Routes: {routes}")

    assert routes == [
        ("/test_api/api/v1/", "GET"),
        ("/test_api/api/v1/error", "GET"),
        ("/test_api/api/v1/query_defaults", "GET"),
        ("/test_api/api/v1/things", "GET"),
        ("/test_api/api/v1/things/{thing_key}", "POST,GET"),
    ]


def test_post_thing():

    _logger.info(f"Type of JSONResponse: {JSONResponse}")

    thing_data = ThingsData(thing_key="test_thing", thing_data={"key": "value"})
    event = {
        "httpMethod": "POST",
        "path": "/test_api/api/v1/things/test_thing",
        "queryStringParameters": {},
        "headers": {},
        "body": json.dumps(thing_data.model_dump(mode="json")),
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 201

    assert response["headers"]["Location"] == "/test_thing"

    body_text = response["body"]

    body_obj = json.loads(body_text)

    assert body_obj["thing_key"] == "test_thing"
    assert body_obj["thing_data"] == {"key": "value"}


def test_get_thing_with_model():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/things/test_thing",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 200

    body_text = response["body"]

    body_obj = json.loads(body_text)

    assert body_obj["thing_key"] == "test_thing"
    assert body_obj["thing_data"] == {"key": "value"}


def test_get_list_of_thigs():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/things",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 200

    body_text = response["body"]

    body_obj = json.loads(body_text)

    _logger.info(f"Body object: {body_obj}")

    assert body_obj == [
        {"thing_key": "one_thing", "thing_data": {"key": "value"}},
        {"thing_key": "another_thing", "thing_data": {"key": "value"}},
    ]


def test_error():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/error",
        "queryStringParameters": {"desired_status_code": "503"},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 503

    body_text = response["body"]

    body_obj = json.loads(body_text)

    assert body_obj["detail"] == "error"
