# idea from http://eyalarubas.com/python-subproc-nonblock.html
from queue import Queue, Empty
from threading import Thread


class NonBlockingStreamReader:
    def __init__(self, stream):
        self._stream = stream
        self._queue = Queue()
        self._thread = None

    def start(self):
        def _worker():
            while True:
                line = self._stream.readline()
                if line:
                    self._queue.put(line)
                else:
                    raise RuntimeError("line is empty")

        self._thread = Thread(target=_worker)
        self._thread.setDaemon(True)
        self._thread.setName("NonBlockingStreamReader of %s" % repr(self._stream))
        self._thread.start()

    def readline(self, timeout=None):
        try:
            return self._queue.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None
