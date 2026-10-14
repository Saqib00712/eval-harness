"""
Phase 28 (Day 17's security review): shared resource ceilings for
untrusted file input. A loaded dataset is text someone might download
from a stranger, generate with a buggy script, or receive from a
teammate who made a mistake -- not output this harness always controls
itself. Without a ceiling, a single oversized or malformed file can hang
a process or exhaust its memory before any of this harness's own
validation ever gets a chance to report a clean error.

This is the concrete fulfillment of the promise `loader.py` made back on
Day 1 ("full enforcement lands in the Phase 28 security day") -- these
three checks (file size, record count, per-record size) are applied by
`loader.py` and `dataset_io.py`, the two places this harness reads
dataset content from disk.

Defaults are GENEROUS, not paranoid: they exist to turn "the process
hangs or runs out of memory" into a clear, named error, not to reject
realistic datasets -- Day 2's own 20-row sample dataset is nowhere close
to any of these. Every limit is an explicit keyword argument on the
loading functions that use it, overridable per call for a caller who
genuinely has a bigger (or smaller) threat model than the default assumes.
"""

DEFAULT_MAX_FILE_SIZE_BYTES = 50_000_000  # 50 MB
DEFAULT_MAX_RECORDS = 100_000
DEFAULT_MAX_RECORD_SIZE_BYTES = 1_000_000  # 1 MB for a single test case / record
