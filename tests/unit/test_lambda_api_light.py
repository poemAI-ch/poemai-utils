import json
import logging
from typing import Optional

from build.lib.poemai_utils.aws.lambda_api_light import JSONResponse
from poemai_utils.aws.lambda_api_light import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    LambdaApiLight,
    Query,
    RedirectResponse,
    Request,
    StreamingResponse,
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


def verify_auth(authorization: Optional[str] = Header(None)):

    if authorization is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if authorization:
        token = authorization[7:]  # Remove "Bearer "

    return {"token": token}


@root_router.get("/")
async def get_root(
    implicit_query_no_default: Optional[str],
    application_service: Application = Depends(get_application_service),
    x_user_id: str = Header(None),
    x_required_header: str = Header(...),
    the_query: str = Query(None),
    implicit_query: Optional[str] = None,
):
    return {
        "message": f"Welcome to the api",
        "available_thing_keys": application_service.get_available_thing_keys(),
        "user_id": x_user_id,
        "query": the_query,
        "required_header": x_required_header,
        "implicit_query": implicit_query,
        "implicit_query_no_default": implicit_query_no_default,
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


@root_router.get("/protected")
async def get_protected(current_user: dict = Depends(verify_auth)):
    return {"auth_info": current_user}


@root_router.get("/cognito/logout_completed")
async def cognito_logout_completed():
    """
    Simulates a Cognito logout endpoint.
    Deletes the session cookie and redirects to the logged-out page.
    """
    logged_out_url = "/logged_out.html"  # Simulate redirect URL

    response = RedirectResponse(url=logged_out_url)
    response.delete_cookie("session_token", path="/")  # Remove session cookie

    return response


@root_router.get("/stream")
async def get_stream():
    return StreamingResponse(
        generator=lambda: ["some", "data", "to", "stream"],
        media_type="application/octet-stream",
        status_code=200,
    )


@root_router.api_route("/proxy/{path:path}", methods=["POST", "GET", "PUT", "DELETE"])
def proxy_request(request: Request, path: str):
    _logger.info(f"Proxying request to path: {path}")
    _logger.info(f"Request object: {request}")
    return JSONResponse(
        status_code=200,
        content={
            "path": path,
            "method": request.method,
            "headers": dict(request.headers),
            "query_params": request.query_params,  # Add query parameters to response
            "body": request.body(),
        },
    )


@root_router.get("/redirect")
async def get_redirect():
    return RedirectResponse(url="/test_api/api/v1/")

@root_router.get("/enum_query")
async def get_enum_query(enum_param: TestEnum = Query(...)):
    return {"enum_value": enum_param.value}

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
        (
            "/test_api/api/v1/cognito/logout_completed",
            "GET",
        ),
        ("/test_api/api/v1/error", "GET"),
        (
            "/test_api/api/v1/protected",
            "GET",
        ),
        (
            "/test_api/api/v1/proxy/{path:path}",
            "DELETE,GET,POST,PUT",
        ),
        ("/test_api/api/v1/query_defaults", "GET"),
        (
            "/test_api/api/v1/redirect",
            "GET",
        ),
        (
            "/test_api/api/v1/stream",
            "GET",
        ),
        ("/test_api/api/v1/things", "GET"),
        ("/test_api/api/v1/things/{thing_key}", "GET,POST"),
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

    _logger.info(f"Response: {response}")
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


def test_proxy():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/proxy/test_path",
        "queryStringParameters": {"param1": "value1", "param2": "value2"},
        "headers": {"x-test-header": "test_value"},
        "body": "request_body_text",
    }

    response = app.handle_request(event, None)
    assert response["statusCode"] == 200

    body_text = response["body"]
    body_obj = json.loads(body_text)

    assert body_obj["path"] == "test_path"
    assert body_obj["method"] == "GET"
    assert (
        body_obj["headers"]["x-test-header"] == "test_value"
    )  # Ensure headers are correct

    # Ensure query parameters are properly passed in the request object
    assert body_obj.get("query_params") == {"param1": "value1", "param2": "value2"}

    assert body_obj["body"] == "request_body_text"


def test_protected_route():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/protected",
        "queryStringParameters": {},
        "headers": {"Authorization": "Bearer test_token"},
        "body": None,
    }

    response = app.handle_request(event, None)
    assert response["statusCode"] == 200

    body_text = response["body"]
    body_obj = json.loads(body_text)

    assert body_obj["auth_info"]["token"] == "test_token"


def test_no_auth():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/protected",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 401


def test_redirect():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/redirect",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 307
    assert response["headers"]["Location"] == "/test_api/api/v1/"


def test_enum_query():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/enum_query",
        "queryStringParameters": {"enum_param": "value2"},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)
    assert response["statusCode"] == 200

    body_text = response["body"]
    body_obj = json.loads(body_text)

    assert body_obj["enum_value"] == "value2"

    # Test with invalid enum value
    event["queryStringParameters"]["enum_param"] = "invalid_value"
    response = app.handle_request(event, None)
    assert response["statusCode"] == 400  # Bad Request


def test_implicit_query():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "implicit_query": "implicit_query_value",
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
    parsed_body = json.loads(body_text)

    assert parsed_body["implicit_query"] == "implicit_query_value"


def test_redirect_with_cookie_deletion():
    """
    Tests whether the logout route properly removes a session cookie before redirecting.
    """
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/cognito/logout_completed",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    _logger.info(f"Response: {response}")

    # Validate redirect response
    assert response["statusCode"] == 307
    assert response["headers"]["Location"] == "/logged_out.html"  # Redirect target

    # Validate that the session cookie is marked for deletion
    assert "Set-Cookie" in response["headers"]

    set_cookie_header = response["headers"]["Set-Cookie"]
    assert "session_token=deleted" in set_cookie_header
    assert "Max-Age=0" in set_cookie_header
    assert "Expires=Thu, 01 Jan 1970 00:00:00 GMT" in set_cookie_header
    assert "Path=/" in set_cookie_header  # Ensure correct cookie path


def test_streaming_response():
    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/stream",
        "queryStringParameters": {},
        "headers": {},
        "body": None,
    }

    response = app.handle_request(event, None)

    assert response["statusCode"] == 200

    body_text = response["body"]
    assert body_text == "somedatatostream"
