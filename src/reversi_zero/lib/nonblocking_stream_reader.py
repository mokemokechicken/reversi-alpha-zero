# idea from http://eyalarubas.com/python-subproc-nonblock.html
from queue import Queue, Empty
from threading import Thread

from logging import getLogger
logger = getLogger(__name__)


class NonBlockingStreamReader:
    def __init__(self, stream):
        self._stream = stream
        self._queue = Queue()
        self._thread = None
        self.closed = True

    def start(self, push_callback=None):
        def _worker():
            while True:
                line = self._stream.readline()
                if line:
                    if push_callback:
                        push_callback(line)
                    self._queue.put(line)
                else:
                    logger.debug("the stream may be closed")
                    break
            self.closed = True

        self._thread = Thread(target=_worker)
        self._thread.setDaemon(True)
        self._thread.setName("NonBlockingStreamReader of %s" % repr(self._stream))
        self.closed = False
        self._thread.start()

    def readline(self, timeout=None):
        try:
            return self._queue.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None
