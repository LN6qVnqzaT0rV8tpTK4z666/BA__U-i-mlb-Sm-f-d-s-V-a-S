# BA__Projekt/BA__Programmierung/util/singleton.py


class Singleton(type):
    """
    A metaclass implementing the Singleton pattern.

    Ensures only one instance of any class using this metaclass exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
