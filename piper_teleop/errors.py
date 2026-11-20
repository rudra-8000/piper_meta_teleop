"""Exceptions shared by the robot API, the server and the client (kept dependency-free so a client machine can import them)."""


class IKError(RuntimeError):
    """The requested tool pose cannot be reached (or only inaccurately)."""
