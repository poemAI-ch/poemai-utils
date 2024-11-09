import json
import logging

from poemai_utils.aws.lambda_api_light import (
    APIRouter,
    Depends,
    Header,
    LambdaApiLight,
    Query,
)

_logger = logging.getLogger(__name__)

_application_service = None


class Application:
    def get_available_corpus_keys(self):
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


PREFIX = f"/poemai_town_bot/api/v1"
router = APIRouter(prefix=PREFIX)
root_router = APIRouter(prefix=PREFIX)


@root_router.get("/")
async def get_root(
    application_service: Application = Depends(get_application_service),
    x_user_id: str = Header(None),
    the_query: str = Query(None),
):
    return {
        "message": f"Welcome to the PoemAI Town Bot API.",
        "available_corpus_keys": application_service.get_available_corpus_keys(),
        "user_id": x_user_id,
        "query": the_query,
    }


app.include_router(root_router)


def test_handle():

    # method = event.get("httpMethod")
    # path = event.get("path")
    # query_params = event.get("queryStringParameters") or {}
    # headers = event.get("headers") or {}
    # body = event.get("body")
    event = {
        "httpMethod": "GET",
        "path": "/poemai_town_bot/api/v1/",
        "queryStringParameters": {
            "corpus_key": "test_corpus",
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

    assert body_obj["message"] == "Welcome to the PoemAI Town Bot API."
    assert body_obj["available_corpus_keys"] == ["test1", "test2", "test3"]
    assert body_obj["user_id"] == "test_user_id"
    assert body_obj["query"] == "test_query"
