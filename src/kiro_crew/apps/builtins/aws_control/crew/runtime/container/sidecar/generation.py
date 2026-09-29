"""The committed-generation pointer, read the same way by both processes.

Each cycle publishes its two authority files as a PAIR into a WRITER-UNIQUE generation --
``gen/<id>/session_map.json`` and ``gen/<id>/open_slots.json``, where ``<id>`` is minted
fresh by :func:`keys.new_generation_id` and never rewritten. A single object -- the
pointer -- names the generation id whose pair is committed, and the commit is a
compare-and-swap on that one object. Three consequences follow, and they are the whole
reason the protocol has this shape:

* Two writers racing in the task-replacement window each mint a DISTINCT id and write a
  distinct generation, so neither overwrites the other's pair and no committed pair is a
  cross-writer tear. The compare-and-swap on the pointer settles which one generation is
  the committed one.
* A cycle interrupted between the pair's two PUTs damages only its own generation, which
  no pointer references. The pointer still names the previous generation, whose objects
  are immutable and were never rewritten, so a replacement boots from a coherent older
  pair rather than a torn newer one.
* Commitment is a single object, and a single object is either there or it is not. There
  is no state in which half a commitment is visible.

The pointer's ABSENCE is meaningful rather than an error. A bucket written before this
protocol holds the authority objects at their ``data/`` keys with no pointer, and that is
read as GENERATION 0. Those objects are never deleted, moved or rewritten, so a bucket
does not have to be migrated to be read, and a writer that predates the protocol keeps
producing buckets this one understands.

The interpretation lives here and neither process keeps its own copy, for the reason
``keys.py`` gives for key derivation: the writer and the reader agreeing with each other
while both disagree with the contract is the failure this shape makes unrepresentable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..common import Settings, keys
from ..common.config import MAX_OBJECT_BYTES
from .store import ObjectAbsent, ObjectStore, StoreUnusable

log = logging.getLogger("smc.sidecar.generation")

__all__ = [
    "Pointer",
    "PointerUnusable",
    "read_pointer",
    "pointer_body",
    "TranscriptIndexUnusable",
    "read_transcript_index",
]


class PointerUnusable(RuntimeError):
    """The pointer is present and cannot be used.

    Distinct from absent, and the distinction decides a boot. Absent means no generation
    has been committed, so the legacy keys are generation 0 and a task starts from them.
    Present-but-unusable means a generation may well be committed and this task cannot
    tell which -- reading that as absence would boot from objects the pointer was steering
    away from.
    """


class TranscriptIndexUnusable(RuntimeError):
    """The committed generation's transcript index is present and cannot be trusted.

    Distinct from absent for the same reason :class:`PointerUnusable` is: an absent index
    (generation 0, or a generation that committed no transcripts) means the front falls
    back to the legacy per-stem key, while an index that is present and will not parse means
    the mapping this task needs to find a blob exists and cannot be read -- and reading a
    stem's blob from the wrong place, or from a legacy key a newer writer never wrote, would
    serve an empty or stale history. So it is refused rather than treated as absent.
    """


@dataclass(frozen=True)
class Pointer:
    """The committed generation: which generation id, and which files it was committed with."""

    generation: str
    authority: frozenset[str]
    #: The INCARNATION of the task that committed this pointer: a lexically-sortable token
    #: minted once per sidecar process (:func:`keys.new_incarnation`), strictly greater for a
    #: later-started task. It fences a SUPERSEDED task from rolling the committed generation
    #: backward. The compare-and-swap on the ETag below rejects a commit whose read of the
    #: pointer went stale, but it cannot catch one overlap: a predecessor whose cycle BEGAN
    #: AFTER its replacement committed reads the replacement's pointer, so it holds a FRESH
    #: ETag and its ``If-Match`` would succeed, publishing its older local-only state over the
    #: replacement's. :func:`~..backup._commit_generation` refuses to commit when the committed
    #: incarnation is strictly newer than its own, so that predecessor steps aside. A pointer
    #: written before this field carries ``""``, which never compares as newer -- a bucket an
    #: earlier writer made stays committable.
    incarnation: str = ""
    #: The object's ETag when this pointer was read, or ``None`` when the store could not
    #: supply one. It is the compare-and-swap validator the commit re-presents as
    #: ``If-Match``: a writer that advanced the pointer since this read changes the ETag, so a
    #: stale commit is rejected rather than overwriting. ``None`` is a MISSING validator, and
    #: the commit fails CLOSED on it -- committing unconditionally would defeat the guard.
    #:
    #: The ETag and the incarnation are two independent fences and BOTH are required: the
    #: ETag catches a concurrent write to the pointer object, and the incarnation catches a
    #: superseded task whose stale commit the ETag check alone would admit (it read a fresh
    #: pointer). The objects they bless are immutable and writer-unique -- a pair at
    #: ``gen/<id>/``, a transcript at ``data/blob/<digest>`` -- so there is no mutable object
    #: a slower writer could tear; what remains to settle is only WHICH already-written
    #: generation the pointer names, and by WHICH task.
    etag: str | None = None


def pointer_body(generation_id: str, incarnation: str = "") -> bytes:
    """The pointer's bytes for a commitment of *generation_id* by task *incarnation*.

    One function so the writer's bytes and the reader's expectations cannot drift; the
    reader's own parsing is the other half and lives in :func:`read_pointer`. *incarnation*
    is the committing task's freshness token, carried so a later read can refuse a commit
    from a task a replacement has superseded.
    """
    return json.dumps(
        {
            "generation": generation_id,
            "incarnation": incarnation,
            "authority": sorted(keys.AUTHORITY_NAMES),
        },
        sort_keys=True,
    ).encode("utf-8")


def read_pointer(settings: Settings, store: ObjectStore) -> Pointer | None:
    """The committed generation, or ``None`` when no pointer has been published.

    ``None`` is the generation-0 answer: the bucket either holds the legacy authority keys
    or holds nothing at all, and both are states a task may boot from.

    Raises :class:`PointerUnusable` for every other way this can go -- a read that fails,
    bytes that do not parse, a slot this writer does not publish into, a missing file list.
    A pointer that exists and cannot be trusted is not permission to look elsewhere.
    """
    key = keys.authority_pointer_key(settings)
    try:
        raw, etag = store.get_with_etag(key, limit=MAX_OBJECT_BYTES)
    except ObjectAbsent:
        return None
    except StoreUnusable:
        # Not translated: the bucket itself cannot be read, so every later read meets the
        # same answer and the process must end on it rather than report a pointer problem.
        raise
    except Exception as exc:  # noqa: BLE001 - translated, never swallowed
        raise PointerUnusable(
            f"the generation pointer could not be read from the bucket ({exc}). This is "
            "not the same as it being absent: absent means no generation is committed and "
            "the legacy keys are this bucket's truth, while unreadable means one may be "
            "committed and this task cannot tell which."
        ) from exc
    # ``etag`` rode the SAME ``get_object`` as the bytes (one GET), so it is the validator for
    # exactly the pointer this cycle read -- no second HEAD a concurrent commit could slip
    # between (which would let a stale ``If-Match`` pass) and nothing unbudgeted for the final
    # cycle's drain window to be killed inside. A store that supplies no ETag leaves it None,
    # and the commit fails CLOSED on a missing validator.
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PointerUnusable(
            f"the generation pointer in the bucket does not parse ({exc}), so it cannot "
            "say which generation is committed."
        ) from exc
    if not isinstance(parsed, dict):
        raise PointerUnusable(
            f"the generation pointer in the bucket is a JSON {type(parsed).__name__}, not "
            "an object, so it names no generation."
        )
    generation_id = parsed.get("generation")
    if not keys.is_generation_id(generation_id):
        raise PointerUnusable(
            f"the generation pointer names generation {generation_id!r}, which is not a "
            "generation id this writer mints (a zero-padded nanosecond timestamp, a hyphen "
            "and hex). No cycle of this writer published it, so the objects it points at are "
            "not a generation this task can read."
        )
    assert isinstance(generation_id, str)  # narrowed by is_generation_id above
    # The task incarnation. A pointer written before this field existed has no 'incarnation'
    # key, read as "" -- the same tolerance the missing-but-newer-field rules below rely on,
    # and "" never compares as newer than a real token, so a bucket an earlier writer made
    # stays committable. A PRESENT incarnation must be a well-formed token (``new_incarnation``
    # mints the generation-id grammar: 20-digit ns, a hyphen, hex). The commit fence compares
    # incarnations LEXICALLY, and a non-conforming non-empty value -- a bare ``"z"``, say --
    # sorts GREATER than every real token, so it would make ``committed.incarnation > this``
    # true for every future task and freeze the commit permanently, restoring stale history.
    # So it is validated by the SAME shape check that guards the generation id (this function
    # exists to distrust bucket bytes), unusable rather than coerced. "" is the one accepted
    # non-token value, for the back-compat reason above.
    raw_incarnation = parsed.get("incarnation", "")
    if raw_incarnation != "" and not keys.is_generation_id(raw_incarnation):
        raise PointerUnusable(
            f"the generation pointer carries incarnation {raw_incarnation!r}, which is "
            "neither empty nor a well-formed task incarnation (a zero-padded nanosecond "
            "timestamp, a hyphen and hex). A malformed value would freeze the commit fence "
            "permanently, so the pointer is unusable rather than trusted."
        )
    listed = parsed.get("authority")
    if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
        raise PointerUnusable(
            "the generation pointer has no 'authority' list of names, so it cannot say "
            "which files the committed generation contains."
        )
    named = frozenset(listed)
    missing = [name for name in keys.AUTHORITY_NAMES if name not in named]
    if missing:
        raise PointerUnusable(
            f"the generation pointer commits a generation without {', '.join(missing)}, "
            "and this writer commits the authority pair whole or not at all. Read as a "
            "partial generation it would present a name this task knows as legitimately "
            "absent, and the backend would flush its own empty view over it -- so the "
            "pointer is unusable rather than a generation missing a member."
        )
    # Names this version does not know are dropped, not refused: a bucket written by a
    # newer writer that commits a third authority file still names a generation whose
    # pair this one can read, and refusing it would make a rollback unbootable. The
    # check above is what keeps that tolerance from also admitting a pointer that
    # under-lists a name this version DOES know.
    return Pointer(
        generation=generation_id,
        authority=frozenset(n for n in named if n in keys.AUTHORITY_NAMES),
        incarnation=raw_incarnation,
        etag=etag,
    )


def read_transcript_index(
    settings: Settings, store: ObjectStore, generation_id: str
) -> dict[str, str]:
    """The committed generation's ``stem -> blob-digest`` map. Never treats absence as empty.

    Called only with a COMMITTED generation id, and every committing cycle writes this index
    object before it commits the pointer -- an EMPTY index when the generation has no live
    transcript, never no object at all. So an object that is ABSENT here is one that external
    retention DELETED while the committed pointer still references it, not a generation that
    legitimately published none. Treating that absence as ``{}`` would read every stem as a
    fresh conversation and the next cycle would overwrite the real history with no recovery
    -- so :class:`ObjectAbsent` is refused as :class:`TranscriptIndexUnusable` rather than
    returning empty. A generation-0 bucket (no committed pointer) never reaches here: the
    caller resolves that before asking for an index.

    Raises :class:`TranscriptIndexUnusable` when the object is present and will not parse,
    or maps a stem to something that is not a well-formed blob digest -- because a value that
    is not a digest cannot be turned into a blob key, and guessing past it would send the
    front to the wrong object. A read that fails for a transport reason is unusable too; a
    denial is never read as absence, for the reason the pointer read gives.

    The values are validated to be blob digests HERE so the front, which turns them into
    keys, cannot be handed a value that carries a path -- the same containment the pointer's
    generation-id check provides.
    """
    key = keys.transcript_index_key(settings, generation_id)
    try:
        raw = store.get(key, limit=MAX_OBJECT_BYTES)
    except ObjectAbsent as exc:
        raise TranscriptIndexUnusable(
            f"the transcript index for the committed generation {generation_id} is absent. "
            "Every committing cycle writes this object (empty when there is no transcript) "
            "before it commits the pointer, so an absent index under a committed pointer is "
            "one retention deleted while it was still referenced -- not a generation that "
            "published none. Reading it as empty would serve every conversation's history as "
            "a fresh one and overwrite it, so the task refuses rather than treat absence as "
            "empty."
        ) from exc
    except StoreUnusable:
        raise
    except Exception as exc:  # noqa: BLE001 - translated, never swallowed
        raise TranscriptIndexUnusable(
            f"the transcript index for generation {generation_id} could not be read "
            f"({exc}). This is not the same as it being absent, so the front refuses to "
            "serve a stale or empty history rather than fall back to a legacy key a newer "
            "writer never wrote."
        ) from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TranscriptIndexUnusable(
            f"the transcript index for generation {generation_id} does not parse ({exc})."
        ) from exc
    if not isinstance(parsed, dict):
        raise TranscriptIndexUnusable(
            f"the transcript index for generation {generation_id} is a JSON "
            f"{type(parsed).__name__}, not an object mapping stems to blob digests."
        )
    for stem, digest in parsed.items():
        if not isinstance(stem, str) or not keys.is_blob_digest(digest):
            raise TranscriptIndexUnusable(
                f"the transcript index for generation {generation_id} maps {stem!r} to "
                f"{digest!r}, which is not a sha256 blob digest. A value that is not a "
                "digest cannot name a blob, and steering a fetch past it could serve the "
                "wrong conversation, so the index is refused rather than partially read."
            )
    return parsed
