# Changes to upstream Splash files

The app lives in `app/` and edits no upstream file, with one exception: a small
**drain** change in the Splash server. It is needed for correct stop and switch,
because Splash's normal shutdown ends the calls that are running.

## Why

`SIGTERM` (and Ctrl+C) stops the server, and `backend.close()` then answers every
running request with `503 server_shutdown`. This was observed against Splash 1.3.0:
a stream of up to 700 tokens that was running when the process got `SIGTERM` ended early
with `{"error": {"code": "server_shutdown"}}` and no `finish_reason`. With the drain it ran to 700 tokens and `finish_reason: length`.
Checking that no call runs before sending `SIGTERM` does not close the gap: a call can be
accepted between the check and the signal. Admission has to close first.

## What changed

| File | Change |
| --- | --- |
| `server/connections.py` | `HttpAdmission.close()` and a `closed` flag. `acquire()` returns false once closed. Granting and closing share one lock, so a slot is granted before `close()` or never. |
| `server/server.py` | `SIGUSR1` handler that only sets an event (`DRAIN_REQUESTED`). A daemon thread (`drain_then_stop`) waits for it, closes the admission of generation requests, waits until no request is held, then sends `SIGTERM` to the process. `do_POST` answers a refused request with `503 server_draining` (`Retry-After: 1`) instead of `frontend_overloaded` when the admission is closed. `/status` gets `http.draining`. |
| `dev/tests/server/test_drain.py` | Tests for the closed admission and for the whole sequence (held request keeps running and finishes, new request refused, stop signal only after the last one). |

About 40 lines of source. Nothing else changes: control routes (`/health`, `/ready`, `/status`,
`/v1/models`) keep answering while draining; a drain that is never requested has no effect.

## How the app uses it

The app reads `http.draining` from `/status`. Its presence means "this Splash can drain".
Without it (any upstream release, including the Homebrew 1.3.0) the app **never sends SIGUSR1**
(an unpatched Python process would be killed by it) and refuses automatic stop and switch over
the API with `drain_unsupported`. A person can still stop it from the window, after a confirmation
that says calls may be cut.

## Getting a Splash with drain support

The Homebrew package does not have it. Build this fork (`make`, see DEVELOPMENT.md) and set
**Settings → Path override** to its `splash` launcher, or propose the change upstream.

## When merging upstream

* `git merge upstream/main`. Conflicts can only occur in the two server files above.
* After resolving, run `python -m unittest dev.tests.server.test_drain` and the HTTP transport tests.
* If upstream adds its own graceful drain, drop this patch, keep `http.draining` (or map the
  app's `Probe.draining` to the new status field) and delete this file.
