"""
Secure error handling utilities.

This module provides functions for safe error handling that prevents
information leakage through error messages while maintaining detailed
logging for debugging purposes.
"""
import logging
from fastapi import HTTPException
from typing import Optional

logger = logging.getLogger(__name__)


def safe_error_response(
    operation: str,
    exception: Exception,
    status_code: int = 500,
    user_message: Optional[str] = None
) -> HTTPException:
    """
    Log full exception details securely, return safe message to user.

    This function ensures that sensitive information from exceptions
    (SQL queries, file paths, stack traces, etc.) is not exposed to
    end users while maintaining detailed logs for debugging.

    Args:
        operation: Description of what was being attempted (e.g., "CSV processing")
        exception: The caught exception
        status_code: HTTP status code to return
        user_message: Optional custom message for user. If None, generates generic message.

    Returns:
        HTTPException with safe user-facing message

    Example:
        try:
            process_file(path)
        except Exception as e:
            raise safe_error_response(
                operation="file processing",
                exception=e,
                status_code=400,
                user_message="Failed to process file. Please check the format."
            )
    """
    # Log full details for debugging (secure logs only)
    logger.exception(f"Error during {operation}: {exception}")

    # Return generic message to user
    if user_message is None:
        user_message = (
            f"An error occurred during {operation}. "
            "Please try again or contact support if the problem persists."
        )

    return HTTPException(status_code=status_code, detail=user_message)


def safe_validation_error(
    field: str,
    user_message: str,
    exception: Optional[Exception] = None
) -> HTTPException:
    """
    Create a validation error with safe messaging.

    Args:
        field: Field that failed validation
        user_message: User-friendly validation message
        exception: Optional exception to log (not shown to user)

    Returns:
        HTTPException with 400 status code
    """
    if exception:
        logger.warning(f"Validation error for field '{field}': {exception}")

    return HTTPException(
        status_code=400,
        detail=f"{field}: {user_message}"
    )
