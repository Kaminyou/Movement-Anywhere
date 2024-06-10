import abc


class Runner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fetch_data(self):
        return NotImplemented

    @abc.abstractmethod
    def upload_data(self):
        return NotImplemented

    @abc.abstractmethod
    def execute(self):
        return NotImplemented

    @abc.abstractmethod
    def clear(self):
        return NotImplemented

    def run(self):
        self.fetch_data()
        self.execute()
        self.upload_data()
        self.clear()
