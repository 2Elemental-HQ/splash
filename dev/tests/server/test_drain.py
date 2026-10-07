import json
import threading
import time
from unittest import mock

from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase, Plan, chat_body
from server import server as api


class DrainTests(HarnessTestCase):
    def test_closed_admission_refuses_new_slots_and_keeps_held_ones(self):
        admission = api.HttpAdmission(2)
        self.assertTrue(admission.acquire())
        admission.close()
        self.assertFalse(admission.acquire())
        self.assertEqual(admission.stats(), {"active": 1, "capacity": 2})
        admission.release()
        self.assertFalse(admission.acquire())

    def test_a_draining_server_finishes_held_requests_and_refuses_new_ones(self):
        blocking = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(blocking), queue_size=4)
        connection, response = harness.open_stream(
            "/v1/chat/completions", chat_body(stream=True)
        )
        self.assertTrue(blocking.started.wait(1))
        self.assertFalse(
            json.loads(harness.request("GET", "/status")[2])["http"]["draining"]
        )

        stopped = threading.Event()
        api.DRAIN_REQUESTED.clear()
        with mock.patch.object(api.os, "kill", lambda pid, number: stopped.set()):
            watcher = threading.Thread(
                target=api.drain_then_stop,
                args=(harness.server,),
                kwargs={"poll": 0.01},
            )
            watcher.start()
            api.DRAIN_REQUESTED.set()
            deadline = time.monotonic() + 2
            while not harness.server.requests.closed and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(harness.server.requests.closed)

            # A new call is refused with its own code; control routes still answer.
            status, _type, payload = harness.request(
                "POST", "/v1/chat/completions", chat_body()
            )
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(payload)["error"]["code"], "server_draining")
            self.assertEqual(harness.request("GET", "/health")[0], 200)
            snapshot = json.loads(harness.request("GET", "/status")[2])["http"]
            self.assertTrue(snapshot["draining"])
            self.assertEqual(snapshot["requests"]["active"], 1)

            # The held call is untouched and the process is not told to stop yet.
            self.assertFalse(stopped.wait(0.3))
            self.assertFalse(blocking.cancelled.is_set())
            blocking.release.set()
            body = response.read()
            self.assertIn(b"[DONE]", body)
            connection.close()
            self.assertTrue(stopped.wait(2), "the stop signal follows the last request")
            watcher.join(2)
        api.DRAIN_REQUESTED.clear()
