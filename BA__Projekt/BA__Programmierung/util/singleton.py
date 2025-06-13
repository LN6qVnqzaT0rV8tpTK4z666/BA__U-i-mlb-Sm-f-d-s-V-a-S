# BA__Projekt/BA__Programmierung/util/singleton.py


class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance per class."""
        if cls not in cls._instances:
            instance = super(Singleton, cls).__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]