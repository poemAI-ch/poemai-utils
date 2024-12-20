import os
from unittest.mock import MagicMock, patch

from poemai_utils.aws.boto3_proxy import boto3_proxy


def test_original_boto3():

    orig_boto3_session = boto3_proxy.Session()


def test_client():

    os.environ["BOTO3_PROXY_PREFIX"] = "TEST_PREFIX"
    os.environ["TEST_PREFIX_S3_AWS_REGION"] = "test-west-region"
    with patch("poemai_utils.aws.boto3_proxy.get_real_boto3") as mock_get_real_boto3:
        mock_boto3 = MagicMock()
        mock_get_real_boto3.return_value = mock_boto3

        s3 = boto3_proxy.client("s3")

        mock_boto3.client.assert_called_with("s3", region_name="test-west-region")


def test_resource():

    os.environ["BOTO3_PROXY_PREFIX"] = "TEST_PREFIX"
    os.environ["TEST_PREFIX_S3_AWS_REGION"] = "test-west-region"
    with patch("poemai_utils.aws.boto3_proxy.get_real_boto3") as mock_get_real_boto3:
        mock_boto3 = MagicMock()
        mock_get_real_boto3.return_value = mock_boto3

        s3 = boto3_proxy.resource("s3")

        mock_boto3.resource.assert_called_with("s3", region_name="test-west-region")


def test_resource_with_endpoint():

    os.environ["BOTO3_PROXY_PREFIX"] = "TEST_PREFIX"
    os.environ["TEST_PREFIX_S3_AWS_REGION"] = "test-west-region"
    os.environ["TEST_PREFIX_S3_ENDPOINT_URL"] = "http://test-endpoint"
    with patch("poemai_utils.aws.boto3_proxy.get_real_boto3") as mock_get_real_boto3:
        mock_boto3 = MagicMock()
        mock_get_real_boto3.return_value = mock_boto3
        mock_boto3.session.Config.return_value = "TESTCONFIG"

        s3 = boto3_proxy.resource("s3")

        mock_boto3.resource.assert_called_with(
            "s3",
            region_name="test-west-region",
            endpoint_url="http://test-endpoint",
            use_ssl=False,
            config="TESTCONFIG",
        )


def test_session():

    os.environ["BOTO3_PROXY_PREFIX"] = "TEST_PREFIX"
    with patch("poemai_utils.aws.boto3_proxy.get_real_boto3") as mock_get_real_boto3:
        mock_boto3 = MagicMock()
        mock_get_real_boto3.return_value = mock_boto3

        s3 = boto3_proxy.Session(region_name="test-west-region")

        mock_boto3.session.Session.assert_called_with(region_name="test-west-region")
