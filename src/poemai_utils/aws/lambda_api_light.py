import asyncio
import concurrent
import copy
import inspect
import json
import logging
import re
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)
from urllib.parse import urlencode, urlparse

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


def convert_value(value: str, annotation: Any):
    if annotation is inspect.Parameter.empty:
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            enum_value = annotation[value]

            return enum_value
        except KeyError as e:
            raise HTTPException(400, f"Invalid value for enum: {value}")
    if annotation in (int, float, bool):
        return annotation(value)

    return value


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
        status_code: int = 200,
    ):
        self.generator = generator
        self.media_type = media_type
        self.headers = headers or {"Content-Type": media_type}
        self.status_code = status_code

    def to_lambda_response(self):
        # AWS Lambda expects the entire body to be sent at once.
        # For true streaming, consider AWS services like API Gateway WebSockets or AWS App Runner.
        # Here, we'll concatenate all chunks for simplicity.
        body = ""
        for chunk in self.generator():
            body += chunk
        return {
            "statusCode": self.status_code,
            "body": body,
            "headers": self.headers,
        }


class RedirectResponse:
    def __init__(
        self, url: str, status_code: int = 307, headers: Dict[str, str] = None
    ):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.cookies_to_delete = []  # Stores cookies to delete

    def delete_cookie(self, key: str, path: str = "/", domain: Optional[str] = None):
        """
        Marks a cookie for deletion by setting it with an expired date.
        """
        cookie = f"{key}=deleted; Path={path}; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; Secure"
        if domain:
            cookie += f"; Domain={domain}"
        self.cookies_to_delete.append(cookie)

    def to_lambda_response(self):
        """
        Converts this response to an AWS Lambda API Gateway-compatible response.
        """
        headers = {**self.headers, "Location": self.url}

        # If there are cookies to delete, add them to headers
        if self.cookies_to_delete:
            headers["Set-Cookie"] = ", ".join(self.cookies_to_delete)

        return {
            "statusCode": self.status_code,
            "headers": headers,
        }


# Exception Classes
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = int(status_code)
        self.detail = detail


class Depends:
    def __init__(self, dependency: Callable):
        self.dependency = dependency

        # Check if the dependency is a function
        if not inspect.isfunction(dependency):
            raise ValueError("Depends requires a function")


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
        self.annotation = None  # We'll set this later


class Request:
    def __init__(
        self,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, Any],
        body: Any,
    ):
        self.method = method
        self.path = path
        self.query_params = query_params or {}
        self.headers = copy.deepcopy(headers)
        self._body = body

        # Construct a FastAPI-like `request.url` object
        self.url = SimpleNamespace(
            path=self.path,
            query=self._construct_query_string(),
            full_url=self._construct_full_url(),
        )

        # Extract cookies from headers
        self.cookies = {}
        cookie_header = headers.get("cookie")
        if cookie_header:
            del headers["cookie"]
            for cookie in cookie_header.split(";"):
                key, value = cookie.split("=")
                self.cookies[key.strip()] = value.strip()

    def body(self):
        return self._body

    def _construct_query_string(self) -> str:
        """Constructs the query string from query parameters."""
        return urlencode(self.query_params, doseq=True)

    def _construct_full_url(self) -> str:
        """Constructs a full URL representation (without domain)."""
        query_string = self._construct_query_string()
        return f"{self.path}?{query_string}" if query_string else self.path


# Assume these helper functions and classes are already defined:
#   - canoncialize_header_name
#   - Depends, Header, Query, Request


class HandlingFunction:
    """
    A helper class to extract injection metadata from a function signature.

    Attributes:
        func: The callable (route handler or dependency function).
        route_param_names: Parameter names coming from the route (e.g. extracted from the URL path).
        signature: The inspected signature of the function.
        dependencies: A dict mapping parameter names to dependency callables (from Depends).
        header_params: A dict mapping canonical header parameter names to Header instances.
        request_params: A dict mapping parameter names (annotated as Request) to their defaults.
        query_params: A dict mapping parameter names to Query instances.
        body_params: A dict mapping parameter names to their inspect.Parameter objects that are meant for the body.
    """

    def __init__(self, func: Callable, route_param_names: Optional[List[str]] = None):
        self.func = func
        self.route_param_names = route_param_names or []
        self.signature = inspect.signature(func)

        self.dependencies: Dict[str, Callable] = self._extract_dependencies()
        self.header_params: Dict[str, Header] = self._extract_header_params()
        self.request_params: Dict[str, Request] = self._extract_request_params()
        self.query_params: Dict[str, Query] = self._extract_query_params()
        self.body_params: Dict[str, inspect.Parameter] = self._extract_body_params()

    def _extract_dependencies(self) -> Dict[str, Callable]:
        """
        Extracts dependencies based on parameters that have a default instance of Depends.
        """
        dependencies = {}
        for name, param in self.signature.parameters.items():
            if isinstance(param.default, Depends):
                # Record the dependency function itself
                dependencies[name] = param.default.dependency
        return dependencies

    def _extract_header_params(self) -> Dict[str, Header]:
        """
        Extracts header parameters based on parameters with a default instance of Header.
        Uses a canonical header name (lowercase) as the key.
        """
        header_params = {}
        for name, param in self.signature.parameters.items():
            canonical_name = canoncialize_header_name(name)
            if isinstance(param.default, Header):
                header_params[canonical_name] = param.default
        return header_params

    def _extract_request_params(self) -> Dict[str, Request]:
        """
        Extracts request parameters by checking for parameters annotated as Request.
        """
        request_params = {}
        for name, param in self.signature.parameters.items():
            if param.annotation == Request:
                request_params[name] = (
                    param.default
                )  # Could be None or a default instance
        return request_params

    def _extract_query_params(self) -> Dict[str, Query]:
        """
        Extracts query parameters based on:
        - Parameters explicitly using Query(...)
        - Primitive types (str, int, float, bool) without a default
        - Optional primitive types (e.g., Optional[str] -> Union[str, NoneType])
        """
        query_params = {}
        for name, param in self.signature.parameters.items():
            param_type = param.annotation
            param_origin = get_origin(param_type)
            param_args = get_args(param_type)

            # Explicitly marked Query param
            if isinstance(param.default, Query):
                query_param = param.default
                query_param.annotation = param.annotation
                query_params[name] = query_param

            # Handle `Optional[T]` → actually `Union[T, NoneType]`
            elif (
                param_origin is Union
                and len(param_args) == 2
                and type(None) in param_args
            ):
                non_none_type = [arg for arg in param_args if arg is not type(None)][0]
                if non_none_type in (
                    str,
                    int,
                    float,
                    bool,
                ):  # Ensure it's a simple type
                    query_param = Query(None)
                    query_param.annotation = non_none_type
                    query_params[name] = query_param

            # Simple primitive type without default → Implicit query param
            elif param.default == inspect.Parameter.empty and param_type in (
                str,
                int,
                float,
                bool,
            ):
                query_param = Query()
                query_param.annotation = param_type
                query_params[name] = query_param

        return query_params

    def _extract_body_params(self) -> Dict[str, inspect.Parameter]:
        """
        Extracts body parameters as those parameters that are not:
            - Route path parameters
            - Dependencies
            - Header parameters
            - Query parameters
            - Request parameters (based on annotation)
        And whose default is not a Depends, Header, or Query.
        """
        body_params = {}
        for name, param in self.signature.parameters.items():
            if (
                name in self.dependencies
                or canoncialize_header_name(name) in self.header_params
                or name in self.query_params
                or name in self.request_params
                or name in self.route_param_names
            ):
                continue
            # We only consider parameters that are explicitly positional/keyword
            if not isinstance(
                param.default, (Depends, Header, Query)
            ) and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                body_params[name] = param
        return body_params

    def get_injection_kwargs(self, available_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a dictionary of already extracted values (e.g. from the headers, query, and path),
        returns a dictionary of arguments to be injected when calling self.func.

        This method can be extended to also resolve dependencies by calling their functions.
        """
        injection_kwargs = {}
        # Inject direct values (headers, query, path, etc.)
        for name in self.signature.parameters:
            if name in available_values:
                injection_kwargs[name] = available_values[name]

        # For dependencies, we can attempt to resolve them by inspecting their signature
        for dep_name, dep_func in self.dependencies.items():
            dep_sig = inspect.signature(dep_func)
            dep_kwargs = {}
            # For each parameter in the dependency, try to provide a value if available
            for pname in dep_sig.parameters:
                if pname in available_values:
                    dep_kwargs[pname] = available_values[pname]
                elif pname == "request" and "request" in available_values:
                    dep_kwargs["request"] = available_values["request"]
            # Call the dependency function and store its result
            injection_kwargs[dep_name] = dep_func(**dep_kwargs)
        return injection_kwargs

    def __repr__(self):
        return (
            f"HandlingFunction(func={self.func.__name__}, "
            f"dependencies={list(self.dependencies.keys())}, "
            f"header_params={list(self.header_params.keys())}, "
            f"query_params={list(self.query_params.keys())}, "
            f"request_params={list(self.request_params.keys())}, "
            f"body_params={list(self.body_params.keys())})"
        )


class InjectedDependency:
    """
    Wraps a dependency function and precomputes an injection map.

    The injection map maps each parameter name in the dependency function's
    signature to a key that is expected in the available values dictionary.

    For example, if a parameter has a default value of type Header, its lookup
    key is determined from the header alias (or using snake_to_header).
    """

    def __init__(self, dep_func: Callable):
        self.dep_func = dep_func
        # Use HandlingFunction to analyze the dependency function
        self.hf = HandlingFunction(dep_func)
        self.injection_map = self._compute_injection_map()

    def _compute_injection_map(self) -> Dict[str, str]:
        mapping = {}
        for param_name, param in self.hf.signature.parameters.items():
            # For a Request parameter, use the key "request"
            if param.annotation == Request:
                mapping[param_name] = "request"
            # For a Header parameter, use the canonical header name.
            elif isinstance(param.default, Header):
                header_alias = param.default.alias or snake_to_header(param_name)
                mapping[param_name] = canoncialize_header_name(header_alias)
            # For a Query parameter, use its alias if provided, else the parameter name.
            elif isinstance(param.default, Query):
                mapping[param_name] = param.default.alias or param_name
            else:
                # Fallback: use the parameter name directly
                mapping[param_name] = param_name
        return mapping

    def resolve(self, available_values: Dict[str, Any]) -> Any:
        """
        Resolves the dependency by building a kwargs dict based on the injection map
        and then calling the dependency function.
        """
        injection_kwargs = {}
        for param_name, lookup_key in self.injection_map.items():
            if lookup_key in available_values:
                injection_kwargs[param_name] = available_values[lookup_key]
            else:
                # If the parameter has a default value, use it.
                param = self.hf.signature.parameters[param_name]
                _logger.info(f"{param_name}: Param: {param}")
                if param.default != inspect.Parameter.empty:
                    param_default_value = param.default
                    if isinstance(param_default_value, Depends):
                        # Resolve the dependency function
                        param_default_value = param_default_value.dependency()
                    elif isinstance(param_default_value, Header):
                        # Use the default value of the header
                        param_default_value = param_default_value.default
                    elif isinstance(param_default_value, Query):
                        param_default_value = param_default_value.default

                    injection_kwargs[param_name] = param_default_value
                else:
                    raise Exception(
                        f"Missing injection for dependency parameter '{param_name}' (lookup key '{lookup_key}')"
                    )
        _logger.info(
            f"Resolved dependency {self.dep_func.__name__} with kwargs {injection_kwargs}"
        )
        return self.dep_func(**injection_kwargs)


# Route Data Structure
class Route:
    def __init__(
        self,
        methods: List[str],
        path: str,
        handler: Callable,
        dependencies: List[str] = None,
    ):
        self.methods = [method.upper() for method in methods]  # Store multiple methods
        self.path = path
        self.handler = handler
        self.dependencies = dependencies or []

        # Find all parameters, both standard and `{param:path}`
        self.param_names = re.findall(r"{(\w+)(?::path)?}", path)

        # Create a HandlingFunction instance for the handler,
        # passing route parameter names to exclude them from body parameters.
        hf = HandlingFunction(handler, route_param_names=self.param_names)

        self.dependencies = (
            {}
        )  # key: parameter name, value: InjectedDependency instance
        # Wrap each dependency function with InjectedDependency.
        for dep_name, dep_func in hf.dependencies.items():
            self.dependencies[dep_name] = InjectedDependency(dep_func)

        self.header_params = hf.header_params
        self.query_params = hf.query_params
        self.request_params = hf.request_params
        self.body_params = hf.body_params

        # Generate regex:
        #   - `{param}` -> `([^/]+)`
        #   - `{param:path}` -> `(.*)`
        regex_pattern = re.sub(r"{(\w+):path}", r"(.*)", path)  # Match entire remainder
        regex_pattern = re.sub(
            r"{(\w+)}", r"([^/]+)", regex_pattern
        )  # Match single segment

        # Final compiled regex
        self.regex = re.compile(f"^{regex_pattern}$")

    def match(self, method: str, path: str) -> Optional[Dict[str, str]]:
        """Match the route against a request method and path."""
        if method.upper() not in self.methods:  # Check if the method is allowed
            return None
        match = self.regex.match(path)
        if not match:
            return None
        params = match.groups()
        retval = dict(zip(self.param_names, params))
        return retval


# Router Class
class APIRouter:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.routes: List[Route] = []

    def add_route(
        self,
        methods: List[str],
        path: str,
        handler: Callable,
        dependencies: List[Any] = None,
    ):
        full_path = self.prefix + path
        route = Route(methods, full_path, handler, dependencies)
        self.routes.append(route)

    # Decorator Factory
    def route(self, method: str, path: str, dependencies: List[str] = None):
        def decorator(func: Callable):
            self.add_route(method, path, func, dependencies)
            return func

        return decorator

    def api_route(self, path: str, methods: List[str], dependencies: List[str] = None):
        def decorator(func: Callable):
            self.add_route(methods, path, func, dependencies)
            return func

        return decorator

    def get(self, path: str, dependencies: List[str] = None):
        return self.api_route(path, ["GET"], dependencies)

    def post(self, path: str, dependencies: List[str] = None):
        return self.api_route(path, ["POST"], dependencies)

    def put(self, path: str, dependencies: List[str] = None):
        return self.api_route(path, ["PUT"], dependencies)

    def delete(self, path: str, dependencies: List[str] = None):
        return self.api_route(path, ["DELETE"], dependencies)

    def patch(self, path: str, dependencies: List[str] = None):
        return self.api_route(path, ["PATCH"], dependencies)


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
        for handler in self.on_startup_handlers:
            handler()

    def find_route(
        self, method: str, path: str
    ) -> Tuple[Optional[Route], Optional[Dict[str, str]]]:
        """Find a matching route by method and path."""
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
        request_object = Request(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=event.get("body"),
        )

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

            # Include body if present
            if body:
                kwargs["body"] = body

            # Build an available values dictionary.
            # Note that keys should correspond to the ones expected by dependency injection.
            available = {"request": request_object}
            # Insert headers with their canonical keys.
            available.update(headers)
            # Insert query parameters.
            available.update(query_params)
            # Also include any path parameters.
            available.update(path_params or {})

            # Resolve dependencies using the preanalyzed metadata.
            for dep_name, injected_dep in route.dependencies.items():
                kwargs[dep_name] = injected_dep.resolve(available)

            # Extract query parameters and path parameters
            if path_params:
                for k, v in path_params.items():
                    if k not in kwargs:
                        kwargs[k] = v

            # Handle query parameters
            for name, query in route.query_params.items():
                if name in kwargs:
                    continue
                query_name = query.alias or snake_to_query_param(name)
                query_value = query_params.get(query_name)
                if query_value is None:
                    if query.default is ...:
                        raise HTTPException(
                            400, f"Missing required query parameter: {query_name}"
                        )
                    else:
                        query_value = query.default
                if query_value is not None:
                    # Convert the value based on the annotation
                    query_value = convert_value(query_value, query.annotation)
                kwargs[name] = query_value
                _logger.info(f"Query Parameter: {name} = {query_value}")

            # Handle body parameters
            for name, param in route.body_params.items():
                # if name in kwargs:
                #     continue
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
                        if name not in kwargs:
                            kwargs[name] = body
                            _logger.info(f"Body Parameter: {name} = {body}")
                elif param.default != inspect.Parameter.empty:
                    if name not in kwargs:
                        kwargs[name] = param.default
                else:
                    raise HTTPException(400, f"Missing required body parameter: {name}")

            for name, _ in route.request_params.items():
                if name not in kwargs:
                    kwargs[name] = request_object

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
            if result_class_name in [
                "JSONResponse",
                "StreamingResponse",
                "RedirectResponse",
            ]:
                _logger.info(
                    f"Returning JSONResponse or StreamingResponse or RedirectResponse converted to Lambda Response"
                )
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
                method_list = []
                for route in routes:
                    method_list.extend(route.methods)
                method_list = sorted(set(method_list))
                methods = ",".join(method_list)
                route_info.append(SimpleNamespace(path=path, methods=methods))

        return route_info
