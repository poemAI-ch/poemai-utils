import logging
import os

_logger = logging.getLogger(__name__)


def get_boto3_proxy_prefix():
    return os.getenv("BOTO3_PROXY_PREFIX", "POEMAI")


def get_real_boto3():
    import boto3

    return boto3


def get_custom_kwargs(service_name):
    """
    Construct kwargs for boto3 client or resource based on environment variables.
    """

    boto3 = get_real_boto3()

    _logger.debug(f"Getting custom kwargs for {service_name}")

    kwargs = {}
    service_region = os.getenv(
        f"{get_boto3_proxy_prefix()}_{service_name.upper()}_AWS_REGION"
    )
    default_region = os.getenv(f"{get_boto3_proxy_prefix()}_DEFAULT_AWS_REGION")

    # Determine which region to use
    if service_region:
        kwargs["region_name"] = service_region
    elif default_region:
        kwargs["region_name"] = default_region

    # Check for service-specific endpoint URL
    endpoint_url = os.getenv(
        f"{get_boto3_proxy_prefix()}_{service_name.upper()}_ENDPOINT_URL"
    )
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
        kwargs["use_ssl"] = (
            False  # Assume no SSL for custom endpoints; adjust if needed
        )

        if service_name == "s3":
            # S3 requires a custom signature version for non-AWS endpoints
            kwargs["config"] = boto3.session.Config(signature_version="s3v4")
        _logger.debug(f"Using custom endpoint URL for {service_name}: {endpoint_url}")

    # check for service specific credentials
    access_key = os.getenv(
        f"{get_boto3_proxy_prefix()}_{service_name.upper()}_ACCESS_KEY_ID"
    )
    secret_key = os.getenv(
        f"{get_boto3_proxy_prefix()}_{service_name.upper()}_SECRET_ACCESS_KEY"
    )

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        _logger.debug(f"Using custom credentials for {service_name}")
    else:
        access_key = os.getenv(f"{get_boto3_proxy_prefix()}_DEFAULT_AWS_ACCESS_KEY_ID")
        secret_key = os.getenv(
            f"{get_boto3_proxy_prefix()}_DEFAULT_AWS_SECRET_ACCESS_KEY"
        )
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            _logger.debug(
                f"Using {get_boto3_proxy_prefix()} default credentials for {service_name}"
            )
        else:
            _logger.debug(f"No custom credentials found for {service_name}")

    return kwargs


class SessionProxy:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        if name in ["client", "resource"]:
            _logger.debug(f"Creating {name} proxy")
            return lambda *args, **kwargs: getattr(self.session, name)(
                *args, **{**get_custom_kwargs(args[0]), **kwargs}
            )
        else:
            return getattr(self.session, name)


class Boto3Proxy:
    def __getattr__(self, name):
        """
        Delegate attribute access to the boto3 module unless it's 'client' or 'resource' or 'Session'.
        """
        boto3 = get_real_boto3()

        if name in ["client", "resource"]:
            return getattr(self, name)
        elif name == "Session":
            return self.Session
        else:
            return getattr(boto3, name)

    def client(self, service_name, **user_kwargs):
        """
        Proxy to boto3.client that injects custom configurations.
        """

        boto3 = get_real_boto3()

        kwargs = get_custom_kwargs(service_name)
        kwargs.update(user_kwargs)  # User kwargs take precedence
        return boto3.client(service_name, **kwargs)

    def resource(self, service_name, **user_kwargs):
        """
        Proxy to boto3.resource that injects custom configurations.
        """
        boto3 = get_real_boto3()

        kwargs = get_custom_kwargs(service_name)
        kwargs.update(user_kwargs)  # User kwargs take precedence
        return boto3.resource(service_name, **kwargs)

    def Session(self, *args, **kwargs):
        boto3 = get_real_boto3()

        session = boto3.session.Session(*args, **kwargs)
        return SessionProxy(session)


# Replace the default boto3 module methods with the proxy methods
boto3_proxy = Boto3Proxy()
