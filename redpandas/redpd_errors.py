"""
Module for custom RedPandas errors
"""


class RedPandasError(Exception):
    """
    Base RedPandas errors class
    """
    def __init__(self, message: str):
        super().__init__(f"RedPandasError: {message}")


class ValueMismatch(RedPandasError):

    def __init__(self, message: str):
        super().__init__(f"V: {message}")
