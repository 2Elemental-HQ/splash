from __future__ import annotations


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code

    @property
    def retryable(self):
        """Whether the same request may succeed when retried shortly: the
        server is overloaded, restarting its engine or shutting down. It
        reports each with 503, whatever status an API answers it with."""
        return self.status == 503

    def protocol_type(self, anthropic=False):
        if not anthropic:
            return "server_error" if self.status >= 500 else "invalid_request_error"
        return {
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            408: "timeout_error",
            413: "request_too_large",
            429: "rate_limit_error",
            503: "overloaded_error",
            504: "timeout_error",
        }.get(
            self.status, "api_error" if self.status >= 500 else "invalid_request_error"
        )


class RequestValidationError(APIError):
    """Request fields that fail validation, each described as FastAPI
    describes one (field_error). /v1/systemone answers with every one, as
    the FastAPI service it stands in for does; the other APIs with the first
    one's message, as with any invalid request."""

    def __init__(self, details):
        super().__init__(400, details[0]["msg"])
        self.details = list(details)


def field_error(loc, msg, error_type="value_error"):
    """What is wrong with the body field at path `loc`, the body itself when
    `loc` is empty."""
    return {"loc": ["body", *loc], "msg": msg, "type": error_type}


class ConstraintError(Exception):
    """The output grammar rejected a token or has no valid next token."""


class ContextLengthError(APIError):
    def __init__(self, input_tokens, maximum_input_tokens, *, image_tokens_only=False):
        super().__init__(
            400, "prompt exceeds the context window", "context_length_exceeded"
        )
        self.input_tokens = input_tokens
        self.maximum_input_tokens = maximum_input_tokens
        self.image_tokens_only = image_tokens_only


class ErrorDialect:
    """How an API answers errors; this one, OpenAI's, also answers requests
    no other API's route takes."""

    def answer(self, error):
        """The status of the response that answers `error`, the code the
        console logs for it, and its payload."""
        return error.status, error.code, self.payload(error)

    def payload(self, error):
        """`error` as a response or an event stream carries it."""
        return {
            "error": {
                "message": error.message,
                "type": error.protocol_type(),
                "code": error.code,
            }
        }


class AnthropicErrors(ErrorDialect):
    def payload(self, error):
        message = error.message
        if isinstance(error, ContextLengthError):
            message = (
                f"prompt is too long: {error.input_tokens} tokens > "
                f"{error.maximum_input_tokens} maximum input tokens"
            )
            if error.image_tokens_only:
                message += " (image tokens alone; text not yet counted)"
        return {
            "type": "error",
            "error": {"type": error.protocol_type(True), "message": message},
        }


class SystemOneErrors(ErrorDialect):
    """/v1/systemone answers as the FastAPI service it stands in for:
    invalid fields with 422 and what is wrong with each, an overload with
    TypeSafe's 529, and other errors as OpenAI's APIs do."""

    def answer(self, error):
        if isinstance(error, RequestValidationError):
            return 422, "unprocessable_entity", {"detail": error.details}
        status, code, payload = super().answer(error)
        return 529 if status == 503 else status, code, payload


OPENAI_ERRORS = ErrorDialect()
ANTHROPIC_ERRORS = AnthropicErrors()
SYSTEMONE_ERRORS = SystemOneErrors()
