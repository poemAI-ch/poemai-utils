import json
import logging

from build.lib.poemai_utils.aws.lambda_api_light import JSONResponse
from poemai_utils.aws.lambda_api_light import (
    APIRouter,
    Depends,
    Header,
    LambdaApiLight,
    Query,
)
from pydantic import BaseModel


class CorpusData(BaseModel):
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
    the_query: str = Query(None),
):
    return {
        "message": f"Welcome to the api",
        "available_thing_keys": application_service.get_available_thing_keys(),
        "user_id": x_user_id,
        "query": the_query,
    }


@root_router.post("/{thing_key}")
async def post_thing(thing_key: str, thing_data: CorpusData):

    headers = {"Location": f"/{thing_key}"}
    return JSONResponse(
        content=thing_data.model_dump(mode="json"),
        status_code=201,
        headers=headers,
    )


app.include_router(root_router)


def test_handle():

    event = {
        "httpMethod": "GET",
        "path": "/test_api/api/v1/",
        "queryStringParameters": {
            "thing_key": "test_thing",
            "the_query": "test_query",
        },
        "headers": {"X-User-Id": "test_user_id"},
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


def test_routes():

    routes = sorted([(r.path, r.methods) for r in app.routes])

    assert routes == [
        ("/test_api/api/v1/", "GET"),
        ("/test_api/api/v1/{thing_key}", "POST"),
    ]


def test_post_thing():

    _logger.info(f"Type of JSONResponse: {JSONResponse}")

    thing_data = CorpusData(thing_key="test_thing", thing_data={"key": "value"})
    event = {
        "httpMethod": "POST",
        "path": "/test_api/api/v1/test_thing",
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
