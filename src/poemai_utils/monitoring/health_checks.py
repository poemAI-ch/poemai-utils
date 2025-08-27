import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

_logger = logging.getLogger(__name__)


def check_database_health() -> Dict[str, Any]:
    """Check DynamoDB table health"""
    try:
        dynamodb = boto3.resource("dynamodb")
        table_name = os.environ.get("DOCUMENT_DB_TABLE_NAME")

        if not table_name:
            return {"status": "unknown", "error": "DOCUMENT_DB_TABLE_NAME not set"}

        table = dynamodb.Table(table_name)

        # Simple table status check
        table_status = table.table_status

        # Try a simple query to test connectivity
        response = table.scan(Limit=1)

        return {
            "status": "healthy" if table_status == "ACTIVE" else "unhealthy",
            "table_status": table_status,
            "error": None,
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_s3_health() -> Dict[str, Any]:
    """Check S3 bucket access"""
    try:
        s3_client = boto3.client("s3")
        bucket_name = os.environ.get("POEMAI_ARTIFACTS_S3_BUCKET", "poemai-artifacts")

        # Simple head bucket request
        s3_client.head_bucket(Bucket=bucket_name)

        return {"status": "healthy", "bucket": bucket_name, "error": None}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_lambda_dependencies() -> Dict[str, Any]:
    """Check if required environment variables are set"""
    required_vars = ["POEMAI_ENVIRONMENT", "AWS_REGION"]

    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    return {
        "status": "healthy" if not missing_vars else "unhealthy",
        "missing_environment_variables": missing_vars,
        "error": (
            None
            if not missing_vars
            else f"Missing environment variables: {missing_vars}"
        ),
    }


def create_health_endpoint_handler():
    """Factory function to create health check endpoint handler"""

    def health_check_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Health check endpoint handler"""

        health_checks = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": os.environ.get("POEMAI_ENVIRONMENT", "unknown"),
            "function_name": context.function_name if context else "unknown",
            "checks": {
                "database": check_database_health(),
                "s3": check_s3_health(),
                "dependencies": check_lambda_dependencies(),
            },
        }

        # Determine overall health
        overall_healthy = all(
            check["status"] == "healthy" for check in health_checks["checks"].values()
        )

        health_checks["overall_status"] = "healthy" if overall_healthy else "unhealthy"

        # Return appropriate HTTP status
        status_code = 200 if overall_healthy else 503

        return {
            "statusCode": status_code,
            "body": json.dumps(health_checks, indent=2),
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
        }

    return health_check_handler


# Example usage in your Lambda functions:
# Add this to each Lambda that should expose a health endpoint
def add_health_endpoint_to_lambda_api(app):
    """Add health endpoint to your LambdaApiLight app"""

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        health_handler = create_health_endpoint_handler()

        # Mock context for health check
        class MockContext:
            function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")

        result = health_handler({}, MockContext())

        # Return the JSON response
        if result["statusCode"] == 200:
            return json.loads(result["body"])
        else:
            from poemai_utils.aws.lambda_api_light import HTTPException

            raise HTTPException(result["statusCode"], json.loads(result["body"]))

    return app
