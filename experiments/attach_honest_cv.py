"""Re-attach to the spawned honest-CV FunctionCall and fetch its result.

The original launcher used .spawn(), so the run is owned by Modal and
survives the local launcher disconnecting. This reads the persisted call
ID and blocks on .get() until the run finishes, then prints the result.
"""
import json
import sys
import modal

CALL_ID_FILE = "/tmp/champion_call_id.txt"


def main():
    call_id = (sys.argv[1] if len(sys.argv) > 1
               else open(CALL_ID_FILE).read().strip())
    print(f"Attaching to {call_id} ...", flush=True)
    fc = modal.FunctionCall.from_id(call_id)
    try:
        r = fc.get(timeout=0)  # non-blocking peek
        print("RESULT_JSON=" + json.dumps(r), flush=True)
    except modal.exception.OutputExpiredError:
        print("STATUS=expired (result no longer retrievable)", flush=True)
    except TimeoutError:
        print("STATUS=running", flush=True)


if __name__ == "__main__":
    main()
