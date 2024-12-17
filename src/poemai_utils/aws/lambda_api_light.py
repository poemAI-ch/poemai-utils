import asyncio
import concurrent
import inspect
import json
import logging
import re
from collections import defaultdict
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


def _is_basic_type(obj):
    if isinstance(obj, (int, float, str, bool)):
        return True


def _serialize_recursive(obj):
    if _is_basic_type(obj):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _serialize_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_recursive(v) for v in obj]
    return obj


def snake_to_header(name: str) -> str:
    """
    Converts snake_case to Header-Case.

    Args:
        name (str): The snake_case string.

    Returns:
        str: The Header-Case string.
    """
    return "-".join(word.capitalize() for word in name.split("_"))


def canoncialize_header_name(name: str) -> str:
    return name.lower()


def snake_to_query_param(name: str) -> str:
    return name


# Response Classes
class JSONResponse:
    def __init__(
        self, content: Any, status_code: int = 200, headers: Dict[str, str] = None
    ):
        self.status_code = status_code
        self.content = json.dumps(content)
        self.headers = headers or {"Content-Type": "application/json"}

    def to_lambda_response(self):
        _logger.info(
            f"Converting JSONResponse to Lambda Response, status code: {self.status_code}"
        )
        return {
            "statusCode": self.status_code,
            "body": self.content,
            "headers": self.headers,
        }


class StreamingResponse:
    def __init__(
        self,
        generator: Callable,
        media_type: str = "application/json",
        headers: Dict[str, str] = None,
    ):
        self.generator = generator
        self.media_type = media_type
        self.headers = headers or {"Content-Type": media_type}

    def to_lambda_response(self):
        # AWS Lambda expects the entire body to be sent at once.
        # For true streaming, consider AWS services like API Gateway WebSockets or AWS App Runner.
        # Here, we'll concatenate all chunks for simplicity.
        body = ""
        for chunk in self.generator():
            body += chunk
        return {
            "statusCode": 200,
            "body": body,
            "headers": self.headers,
        }


# Exception Classes
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = int(status_code)
        self.detail = detail


class Depends:
    def __init__(self, dependency: Callable):
        _logger.info(f"Constructing Depends with {dependency}")
        self.dependency = dependency


class Header:
    def __init__(self, default: Any = ..., alias: str = None):
        """
        Represents a header parameter.

        Args:
            default (Any, optional): The default value if the header is not provided.
            alias (str, optional): The actual header name in the HTTP request.
        """
        self.default = default
        self.alias = alias


class Query:
    def __init__(self, default: Any = ..., alias: str = None):
        """
        Represents a query parameter.

        Args:
            default (Any, optional): The default value if the query parameter is not provided.
            alias (str, optional): The actual query parameter name in the HTTP request.
        """
        self.default = default
        self.alias = alias


# Route Data Structure
class Route:
    def __init__(
        self, method: str, path: str, handler: Callable, dependencies: List[str] = None
    ):
        self.method = method.upper()
        self.path = path
        self.handler = handler
        self.dependencies = dependencies or []
        # Compile regex for path matching and extract path parameters
        self.param_names = re.findall(r"{(\w+)}", path)
        self.regex = re.compile("^" + re.sub(r"{\w+}", r"([^/]+)", path) + "$")
        self.dependencies = self._extract_dependencies()
        self.header_params = self._extract_header_params()
        self.query_params = self._extract_query_params()
        self.body_params = self._extract_body_params()

    def _extract_dependencies(self) -> Dict[str, Callable]:
        dependencies = {}
        sig = inspect.signature(self.handler)
        for name, param in sig.parameters.items():
            if isinstance(param.default, Depends):
                _logger.info(
                    f"Found dependency for {name} : {param.default.dependency}"
                )

                dependencies[name] = param.default.dependency
        return dependencies

    def _extract_header_params(self) -> Dict[str, Header]:
        header_params = {}
        sig = inspect.signature(self.handler)
        for name, param in sig.parameters.items():
            canonical_name = canoncialize_header_name(name)
            if isinstance(param.default, Header):
                header_params[canonical_name] = param.default
                _logger.info(
                    f"Found header parameter '{canonical_name}' with alias '{param.default.alias}' and default '{param.default.default}'"
                )
        return header_params

    def _extract_query_params(self) -> Dict[str, Query]:
        query_params = {}
        sig = inspect.signature(self.handler)
        for name, param in sig.parameters.items():
            if isinstance(param.default, Query):
                query_params[name] = param.default
                _logger.info(
                    f"Found query parameter '{name}' with alias '{param.default.alias}' and default '{param.default.default}'"
                )
        return query_params

    def _extract_body_params(self) -> Dict[str, Any]:
        body_params = {}
        sig = inspect.signature(self.handler)
        for name, param in sig.parameters.items():
            if (
                name in self.query_params
                or name in self.header_params
                or name in self.dependencies
                or name in self.param_names
            ):
                continue
            if not isinstance(
                param.default, (Depends, Header, Query)
            ) and param.kind in [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ]:
                body_params[name] = param
                _logger.info(f"Found body parameter '{name}'")
        return body_params

    def match(self, method: str, path: str) -> Optional[Dict[str, str]]:
        if self.method != method:
            return None
        match = self.regex.match(path)
        if not match:
            return None
        params = match.groups()
        return dict(zip(self.param_names, params))


# Router Class
class APIRouter:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.routes: List[Route] = []

    def add_route(
        self, method: str, path: str, handler: Callable, dependencies: List[Any] = None
    ):
        full_path = self.prefix + path
        route = Route(method, full_path, handler, dependencies)
        self.routes.append(route)

    # Decorator Factory
    def route(self, method: str, path: str, dependencies: List[str] = None):
        def decorator(func: Callable):
            self.add_route(method, path, func, dependencies)
            return func

        return decorator

    def get(self, path: str, dependencies: List[str] = None):
        return self.route("GET", path, dependencies)

    def post(self, path: str, dependencies: List[str] = None):
        return self.route("POST", path, dependencies)


class LambdaApiLight:
    def __init__(self):
        self.routers: List[APIRouter] = []
        self.event_handlers: Dict[str, List[Callable]] = {}

        self.logger = logging.getLogger("lightweight_framework")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def include_router(self, router: APIRouter):
        self.routers.append(router)

    def on_startup(self, func: Callable):
        self.on_startup_handlers.append(func)
        return func

    async def execute_event_handlers(self, event_type: str):
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            if inspect.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

    def on_event(self, event_type: str):
        """
        Decorator to register an event handler for a specific event type.
        Usage:
            @app.on_event("startup")
            async def startup_event():
                ...
        """

        def decorator(func: Callable):
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(func)
            return func

        return decorator

    def execute_startup(self):
        """
        Executes all startup event handlers.
        """
        asyncio.run(self.execute_event_handlers("startup"))

    def execute_startup(self):
        for handler in self.on_startup_handlers:
            handler()

    def find_route(
        self, method: str, path: str
    ) -> Tuple[Optional[Route], Optional[Dict[str, str]]]:
        for router in self.routers:
            for route in router.routes:
                params = route.match(method, path)
                if params is not None:
                    return route, params
        return None, None

    def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        method = event.get("httpMethod")
        path = event.get("path")
        query_params = event.get("queryStringParameters") or {}
        headers = event.get("headers") or {}
        headers = {canoncialize_header_name(k): v for k, v in headers.items()}

        body = event.get("body")
        is_base64_encoded = event.get("isBase64Encoded", False)

        route, path_params = self.find_route(method, path)
        if not route:
            return JSONResponse(
                {"error": "Not Found"}, status_code=404
            ).to_lambda_response()
        try:
            # Parse body
            if body:
                if is_base64_encoded:
                    import base64

                    body = base64.b64decode(body).decode("utf-8")
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    pass  # Keep as string if not JSON

            # Handle dependencies
            kwargs = {}

            # Include body if present
            if body:
                kwargs["body"] = body

            for name, dep_callable in route.dependencies.items():
                kwargs[name] = dep_callable()

            # Extract query parameters and path parameters
            if path_params:
                kwargs.update(path_params)
            if query_params:
                kwargs.update(query_params)

            # Handle header parameters
            for name, header in route.header_params.items():

                header_name = canoncialize_header_name(
                    header.alias or snake_to_header(name)
                )

                header_value = headers.get(header_name)  # Extract the header value

                if header_value is None:  # Header not provided
                    if header.default is ...:  # Explicitly required header
                        raise HTTPException(
                            400, f"Missing required header: {header_name}"
                        )
                    else:  # Optional header with a default value
                        header_value = header.default

                kwargs[name] = header_value  # Populate the route function parameters
                _logger.info(f"Header Parameter: {name} = {header_value}")

            # Handle query parameters
            for name, query in route.query_params.items():
                query_name = query.alias or snake_to_query_param(name)
                query_value = query_params.get(query_name)
                if query_value is None:
                    if query.default is ...:
                        raise HTTPException(
                            400, f"Missing required query parameter: {query_name}"
                        )
                    else:
                        query_value = query.default

                kwargs[name] = query_value
                _logger.info(f"Query Parameter: {name} = {query_value}")

            # Handle body parameters
            for name, param in route.body_params.items():
                if body:

                    if hasattr(param.annotation, "model_validate"):
                        # Assume it's a Pydantic model
                        try:
                            kwargs[name] = param.annotation.model_validate(body)
                            _logger.info(
                                f"Parsed body parameter '{name}' into {param.annotation}"
                            )
                        except Exception as e:
                            _logger.warning(
                                f"Failed to parse body parameter '{name}' into {param.annotation}: {e}"
                            )
                            raise HTTPException(
                                400, f"Invalid body for parameter '{name}': {e}"
                            )
                    else:
                        kwargs[name] = body
                        _logger.info(f"Body Parameter: {name} = {body}")
                elif param.default != inspect.Parameter.empty:
                    kwargs[name] = param.default
                else:
                    raise HTTPException(400, f"Missing required body parameter: {name}")

            sig = inspect.signature(route.handler)
            bound_args = {}
            for name, param in sig.parameters.items():
                if name in kwargs:
                    bound_args[name] = kwargs[name]
                elif param.default != inspect.Parameter.empty:
                    bound_args[name] = param.default
                else:
                    raise Exception(f"Missing required parameter: {name}")

            result = route.handler(**bound_args)

            # Check if the result is a coroutine and handle it
            if inspect.iscoroutine(result):
                try:
                    # Attempt to run the coroutine using asyncio.run
                    result = asyncio.run(result)
                except RuntimeError:
                    # If there's already a running event loop (e.g., during testing), use a new thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(loop.run_until_complete, result)
                        result = future.result()
                    loop.close()
            result_class_name = result.__class__.__name__
            if result_class_name in ["JSONResponse", "StreamingResponse"]:
                _logger.info(f"Returning JSONResponse or StreamingResponse converted")
                return result.to_lambda_response()
            elif isinstance(result, dict):
                _logger.info(f"JSONResponse converting dict to json")
                return JSONResponse(result).to_lambda_response()
            elif isinstance(result, list):
                _logger.info(f"JSONResponse converting list to json")
                return JSONResponse(_serialize_recursive(result)).to_lambda_response()
            elif hasattr(result, "model_dump"):
                _logger.info(f"JSONResponse converting model to json")
                return JSONResponse(result.model_dump(mode="json")).to_lambda_response()
            else:
                _logger.info(f"Returning result as string")
                # Assume string
                return {
                    "statusCode": 200,
                    "body": str(result),
                    "headers": {"Content-Type": "text/plain"},
                }

        except HTTPException as he:
            return {
                "statusCode": he.status_code,
                "body": json.dumps({"detail": he.detail}),
                "headers": {"Content-Type": "application/json"},
            }
        except Exception as e:
            self.logger.error(f"Unhandled exception: {e}", exc_info=True)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Internal Server Error"}),
                "headers": {"Content-Type": "application/json"},
            }

    @property
    def routes(self):
        route_info = []
        for router in self.routers:
            by_path = defaultdict(list)
            for route in router.routes:
                by_path[route.path].append(route)

            for path, routes in by_path.items():
                methods = ",".join(route.method for route in routes)
                route_info.append(SimpleNamespace(path=path, methods=methods))

        return route_info
