"""HTTP ingress bounds: the connections the server serves at once, the
requests and request body bytes it admits, and the closing of connections it
refuses."""

import select
import socket
import threading
import time
import weakref

from . import json_codec
from .errors import APIError

# The answer to a connection that gets no slot, or gives up its slot before
# its request is read. With no request path, no API dialect is known: a
# generic server error with its stable diagnostic code.
_CONNECTION_OVERLOADED_PAYLOAD = (
    b'{"error":{"type":"server_error","code":"frontend_overloaded",'
    b'"message":"HTTP connection capacity is exhausted"}}'
)
CONNECTION_OVERLOADED_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\n"
    b"Content-Length: %d\r\n\r\n%s"
    % (len(_CONNECTION_OVERLOADED_PAYLOAD), _CONNECTION_OVERLOADED_PAYLOAD)
)


class HttpAdmission:
    """Nonwaiting capacity gate, in request counts or input bytes."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("HTTP admission capacity must be a positive integer")
        self.capacity = capacity
        self.active = 0
        # Input finalizers can run during a stats snapshot on this thread.
        self.lock = threading.RLock()

    def acquire(self, amount=1):
        with self.lock:
            if self.active + amount > self.capacity:
                return False
            self.active += amount
            return True

    def release(self, amount=1):
        with self.lock:
            if amount > self.active:
                raise RuntimeError("HTTP admission slot released without acquisition")
            self.active -= amount

    def stats(self):
        with self.lock:
            return {"active": self.active, "capacity": self.capacity}


def refuse_connection(connection):
    """Send CONNECTION_OVERLOADED_RESPONSE without waiting: the server has
    read no request from the connection and written it no response, so the
    response fits in its send buffer."""
    try:
        connection.send(CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT)
    except OSError:
        pass


def _has_input(connection):
    """Whether `connection` holds input its thread has yet to read, such as
    a request that arrived before its thread ran."""
    # A poll object holds no descriptor.
    poller = select.poll()
    poller.register(connection, select.POLLIN)
    return bool(poller.poll(0))


class LingeringCloser:
    """Closes connections answered without reading their requests.

    Closing a connection with request bytes unread resets it, and the reset
    can destroy the answer before the client reads it. Each connection given
    here is half-closed instead; one thread reads and drops what its client
    still sends, and closes it once the client has closed or `linger`
    seconds after its answer. At most `capacity` wait at once; any beyond
    them are closed at once.
    """

    # How soon the thread first reads a connection that arrives while it
    # waits on others.
    TICK = 0.05

    def __init__(self, capacity, linger):
        self.capacity = capacity
        self.linger = linger
        self.changed = threading.Condition()
        self.arrivals = []
        self.held = 0
        self.stopped = False
        self.thread = threading.Thread(
            target=self._run, name="lingering close", daemon=True
        )
        self.thread.start()

    def close(self, connection):
        try:
            connection.shutdown(socket.SHUT_WR)
            connection.setblocking(False)
        except OSError:
            connection.close()
            return
        with self.changed:
            if self.stopped or self.held >= self.capacity:
                connection.close()
                return
            self.held += 1
            self.arrivals.append((connection, time.monotonic() + self.linger))
            self.changed.notify()

    def stop(self):
        """Close every waiting connection and end the thread."""
        with self.changed:
            self.stopped = True
            self.changed.notify()
        self.thread.join()

    def _run(self):
        # A poll object holds no descriptor.
        poller = select.poll()
        waiting = {}
        while True:
            with self.changed:
                while not (waiting or self.arrivals or self.stopped):
                    self.changed.wait()
                arrivals, self.arrivals = self.arrivals, []
                stopped = self.stopped
            for connection, deadline in arrivals:
                poller.register(connection, select.POLLIN)
                waiting[connection.fileno()] = connection, deadline
            if not stopped:
                for descriptor, _ in poller.poll(self.TICK * 1000):
                    connection, _ = waiting[descriptor]
                    try:
                        if connection.recv(65536):
                            continue
                    except BlockingIOError:
                        continue
                    except OSError:
                        pass
                    waiting[descriptor] = connection, 0.0
            now = time.monotonic()
            done = [
                descriptor
                for descriptor, (_, deadline) in waiting.items()
                if stopped or deadline <= now
            ]
            for descriptor in done:
                poller.unregister(descriptor)
                waiting.pop(descriptor)[0].close()
            with self.changed:
                self.held -= len(done)
            if stopped:
                return


class ConnectionSlots:
    """The connections the server gives a thread, at most `capacity`.

    One that waits for its request with nothing yet to read, or drains an
    upload refused unread, gives its slot to a new connection when no slot
    is free, the longest waiting first, so stalled connections, however
    many and from however many addresses, cannot keep others out. One whose
    request has arrived keeps its slot, although its thread may not have
    run yet: when every slot has a request, arrived or in progress, the new
    connection is refused. One that gives way still awaiting its request
    gets the 503 of a connection refused at the accept, as its request may
    be on its way. One draining a refused upload already has its response.
    """

    # What a connection with a slot is doing.
    AWAITING_REQUEST = "awaiting request"
    SERVING = "serving"
    DRAINING = "draining"

    def __init__(self, capacity):
        self.capacity = capacity
        # Each connection with a slot, mapped to what it is doing; those that
        # wait on their client in the order they began to.
        self.holders = {}
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()

    def admit(self, connection):
        """Give `connection` a slot, awaiting its request; False when every
        slot has a request, arrived or in progress."""
        with self.lock:
            if len(self.holders) >= self.capacity:
                waiting = next(
                    (
                        held
                        for held, state in self.holders.items()
                        if state == self.DRAINING
                        or (state == self.AWAITING_REQUEST and not _has_input(held))
                    ),
                    None,
                )
                if waiting is None:
                    return False
                if self.holders[waiting] == self.AWAITING_REQUEST:
                    refuse_connection(waiting)
                self._close(waiting)
            self.holders[connection] = self.AWAITING_REQUEST
            self.idle.clear()
            return True

    def expire(self, connection):
        """Close `connection`, and free its slot, if it still awaits its
        request."""
        with self.lock:
            if self.holders.get(connection) == self.AWAITING_REQUEST:
                self._close(connection)

    def _close(self, connection):
        # Under the lock, which a connection's release takes before the
        # connection is closed, so the descriptor is still its own. Its
        # thread sees the end of input; it has lost its slot, so a request
        # whose headers the shutdown cut short is not served.
        del self.holders[connection]
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def serving(self, connection):
        """Mark the request on `connection` in progress; False once the
        connection has lost its slot."""
        with self.lock:
            if connection not in self.holders:
                return False
            self.holders[connection] = self.SERVING
            return True

    def draining(self, connection):
        """`connection`, answered, drains an upload refused unread: it waits
        on its client again, last in line."""
        with self.lock:
            if self.holders.pop(connection, None) is not None:
                self.holders[connection] = self.DRAINING

    def release(self, connection):
        """Give back the slot of `connection`, if it still has one, before
        the connection is closed."""
        with self.lock:
            self.holders.pop(connection, None)
            if not self.holders:
                self.idle.set()

    def stats(self):
        with self.lock:
            return {"active": len(self.holders), "capacity": self.capacity}


class RequestBodyReservation:
    """Account input bytes until preparation and any retained input are released."""

    def __init__(self, admission, size):
        if not admission.acquire(size):
            raise APIError(
                503,
                "request body capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.admission = admission
        self.size = size

    def release(self):
        self.admission.release(self.size)
        self.size = 0

    def grow(self, size):
        if not self.admission.acquire(size):
            raise APIError(
                503,
                "retained input capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.size += size

    def retain_for(self, job):
        # Generation retains schemas and, for Responses, conversation history.
        # Text/image prompts have otherwise become tokens and prepared pixels.
        policy = job.tool_policy
        retained = (
            job.response_history_items,
            job.response_format,
            policy.schemas if policy else None,
            policy.namespaces if policy else None,
            job.stop_sequences,
        )
        retained = [value for value in retained if value]
        size = json_codec.encoded_size(retained) if retained else 0
        if size > self.size:
            self.grow(size - self.size)
        else:
            self.admission.release(self.size - size)
        self.size = size
        if size:
            weakref.finalize(job, self.release)
