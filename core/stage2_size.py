"""Stage 2 size: B-tree page layout and postgres_btree_tuple_layout_v3."""

import math
import re
from contextlib import contextmanager
from itertools import product

import psycopg2

class PostgresBTreeLayout:
    """Host B-tree page layout. These are PostgreSQL defaults, not our knobs.

    The size model is the textbook three-step estimator. This table only
    instantiates it for PostgreSQL 14's default B-tree (BLCKSZ, fillfactor,
    IndexTuple / ItemId / TID widths, and the v4 max item size). Changing
    the host engine means swapping this layout; the three steps stay the same.
    """

    PAGE_BYTES = 8192
    PAGE_HEADER_BYTES = 40
    LEAF_FILLFACTOR = 90
    INTERNAL_FILLFACTOR = 70
    TID_BYTES = 6
    TUPLE_HEADER_BYTES = 8  # IndexTuple header, includes the heap TID.
    LINE_POINTER_BYTES = 4
    NULL_BITMAP_BYTES = 8  # MAXALIGN of the optional null bitmap.
    # btree version 4: "index row size exceeds ... maximum 2704"
    MAX_ITEM_BYTES = 2704
    # nbtsort.c: a posting list is capped at about 1/10 of a page, MAXALIGN'd.
    POSTING_PAGE_DIVISOR = 10


BTREE = PostgresBTreeLayout

class _SizeSpec:
    def __init__(
        self, source_id, table, keys, estimated_size=0, schema="public", unique=False
    ):
        self.id = source_id
        self.schema = schema
        self.table = table
        self.keys = tuple(keys)
        self.estimated_size = int(estimated_size or 0)
        self.unique = bool(unique)


def _align_bytes(size, boundary=8):
    return int(math.ceil(size / boundary)) * boundary


class SizeCalibrator:
    """Textbook B-tree size model: stats → leaf payload → pages.

    The published estimator is the usual three-box figure, not a bag of
    special cases. Each box has one job; PostgreSQL layout details stay
    inside that box and do not change the outer data flow.

        pg_stats(rows, width, NDV, nulls)
            → leaf payload  (tuple layout; posting if the type can dedup)
            → index bytes   (leaf fillfactor + internal fanout)

    Same inputs, same arithmetic, same ``est_size_bytes``. Catalog
    statistics only; no workload execution and no existing-index sizes.
    Numeric widths and fillfactors live in ``PostgresBTreeLayout`` so the
    algorithm figure does not carry host-specific constants.
    """

    def __init__(self, host, port, user, password, dbname):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.dbname = dbname
        self.conn = None
        self._table_statistics_cache = {}
        self._max_width_cache = {}

    def max_key_bytes(self, table, keys, schema="public"):
        """Largest possible B-tree row for ``keys``: fixed types use pg_stats
        avg_width; unbounded text types are measured with max(pg_column_size)."""
        total = 0
        for key in keys:
            cache_key = (schema, table, key)
            width = self._max_width_cache.get(cache_key)
            if width is None:
                with self.cursor() as cur:
                    cur.execute(
                        """
                        SELECT format_type(a.atttypid, a.atttypmod), a.attlen,
                               COALESCE(s.avg_width, 0)
                        FROM pg_attribute a
                        JOIN pg_class c ON c.oid = a.attrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        LEFT JOIN pg_stats s ON s.schemaname = n.nspname
                             AND s.tablename = c.relname AND s.attname = a.attname
                        WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s
                        """,
                        (schema, table, key),
                    )
                    row = cur.fetchone()
                    if row is None:
                        width = 0
                    elif int(row[1]) > 0:
                        width = int(row[1])
                    else:
                        length = re.search(r"\((\d+)\)", row[0] or "")
                        if length and 4 * int(length.group(1)) <= BTREE.MAX_ITEM_BYTES:
                            width = int(row[2])
                        else:
                            cur.execute(
                                "SELECT COALESCE(max(pg_column_size(%s)), 0) FROM %s.%s"
                                % (
                                    psycopg2.extensions.quote_ident(key, cur),
                                    psycopg2.extensions.quote_ident(schema, cur),
                                    psycopg2.extensions.quote_ident(table, cur),
                                )
                            )
                            width = int(cur.fetchone()[0])
                self._max_width_cache[cache_key] = width
            total += width
        return total

    def connect(self):
        if self.conn is not None and not self.conn.closed:
            return
        kwargs = {
            "port": self.port,
            "user": self.user,
            "dbname": self.dbname,
            "application_name": "candidate-pool-size",
        }
        if self.host:
            kwargs["host"] = self.host
        if self.password:
            kwargs["password"] = self.password
        self.conn = psycopg2.connect(**kwargs)
        self.conn.autocommit = True

    def close(self):
        if self.conn is not None and not self.conn.closed:
            self.conn.close()
        self.conn = None

    @contextmanager
    def cursor(self):
        self.connect()
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def _relation_statistics(self, schema, table):
        """Box 1: catalog inputs. One ``(reltuples, columns)`` snapshot per table."""
        table_key = (schema, table)
        cached = self._table_statistics_cache.get(table_key)
        if cached is not None:
            return cached
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT c.reltuples
                FROM pg_class c
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname=%s AND c.relname=%s
                  AND c.relkind IN ('r','m','p')
                """,
                table_key,
            )
            row = cur.fetchone()
            relation_rows = max(0.0, float(row[0])) if row else 0.0
            cur.execute(
                """
                SELECT s.attname, s.avg_width, s.n_distinct, s.null_frac,
                       a.attlen, a.attalign, a.attstorage,
                       COALESCE(coll.collisdeterministic, true) AND EXISTS (
                           SELECT 1 FROM pg_opclass opc
                           JOIN pg_am am ON am.oid=opc.opcmethod
                           JOIN pg_amproc ap ON ap.amprocfamily=opc.opcfamily
                               AND ap.amproclefttype=opc.opcintype
                               AND ap.amprocrighttype=opc.opcintype
                               AND ap.amprocnum=4
                           JOIN pg_proc proc ON proc.oid=ap.amproc
                           WHERE am.amname='btree' AND opc.opcdefault
                             AND proc.pronamespace='pg_catalog'::regnamespace
                             AND proc.proname IN ('btequalimage', 'btvarstrequalimage')
                             AND (opc.opcintype=a.atttypid OR opc.opcintype IN (
                                 SELECT casttarget FROM pg_cast
                                 WHERE castsource=a.atttypid
                                   AND castmethod='b' AND castcontext='i'))
                       ) AS deduplicable
                FROM pg_stats s
                JOIN pg_namespace n ON n.nspname=s.schemaname
                JOIN pg_class c ON c.relnamespace=n.oid AND c.relname=s.tablename
                JOIN pg_attribute a ON a.attrelid=c.oid AND a.attname=s.attname
                LEFT JOIN pg_collation coll ON coll.oid=a.attcollation
                WHERE s.schemaname=%s AND s.tablename=%s
                """,
                table_key,
            )
            column_stats = {
                str(name): (int(width), float(distinct), float(null_frac),
                            int(length), alignment, storage, bool(deduplicable))
                for name, width, distinct, null_frac, length, alignment,
                    storage, deduplicable in cur.fetchall()
            }
        cached = (relation_rows, column_stats)
        self._table_statistics_cache[table_key] = cached
        return cached

    def _index_tuple_layout(self, columns, nulls, relation_rows):
        """One NULL/non-NULL combination: probability, tuple width, NDV product."""
        probability = 1.0
        offset = BTREE.TUPLE_HEADER_BYTES + (BTREE.NULL_BITMAP_BYTES if any(nulls) else 0)
        distinct_keys = 1.0
        for is_null, column in zip(nulls, columns):
            width, raw_distinct, null_frac, length, alignment, storage, _ = column
            probability *= null_frac if is_null else 1 - null_frac
            if is_null:
                continue
            distinct_keys *= max(
                1.0,
                -raw_distinct * relation_rows if raw_distinct < 0 else raw_distinct,
            )
            boundary = {"c": 1, "s": 2, "i": 4, "d": 8}[alignment]
            # pg_stats.avg_width already includes the stored short header.
            if length == -1 and storage != "p" and width <= 127:
                boundary = 1
            offset = _align_bytes(offset, boundary) + max(1, width)
        tuple_bytes = _align_bytes(offset)
        return probability, tuple_bytes, tuple_bytes + BTREE.LINE_POINTER_BYTES, distinct_keys

    def _posting_payload(self, layout_rows, ordinary_entry_bytes, tuple_bytes, distinct_keys):
        """Optional leaf compression when every key type supports equalimage."""
        max_posting_bytes = (
            (BTREE.PAGE_BYTES // BTREE.POSTING_PAGE_DIVISOR // 8) * 8
            - BTREE.LINE_POINTER_BYTES
        )
        posting_limit = (max_posting_bytes // 8) * 8
        max_tids = int((posting_limit - tuple_bytes) // BTREE.TID_BYTES)
        if max_tids < 2:
            return layout_rows * ordinary_entry_bytes
        distinct_keys = min(layout_rows, distinct_keys)
        tids_per_key = layout_rows / distinct_keys
        full, remainder = divmod(tids_per_key, max_tids)
        group_bytes = full * (
            _align_bytes(tuple_bytes + max_tids * BTREE.TID_BYTES)
            + BTREE.LINE_POINTER_BYTES
        )
        if remainder > 1:
            group_bytes += (
                _align_bytes(tuple_bytes + remainder * BTREE.TID_BYTES)
                + BTREE.LINE_POINTER_BYTES
            )
        else:
            group_bytes += remainder * ordinary_entry_bytes
        return min(layout_rows * ordinary_entry_bytes, distinct_keys * group_bytes)

    def _leaf_payload(self, relation_rows, columns, unique):
        """Box 2: leaf store. Mix NULL layouts; apply posting only when legal."""
        null_layouts = [
            (False,) if column[2] == 0 else (True,) if column[2] == 1
            else (False, True) for column in columns
        ]
        leaf_payload = 0.0
        average_entry_bytes = 0.0
        can_dedup = (not unique) and all(column[-1] for column in columns)
        for nulls in product(*null_layouts):
            probability, tuple_bytes, ordinary_entry_bytes, distinct_keys = (
                self._index_tuple_layout(columns, nulls, relation_rows)
            )
            average_entry_bytes += probability * ordinary_entry_bytes
            layout_rows = relation_rows * probability
            if layout_rows <= 0:
                continue
            if can_dedup:
                leaf_payload += self._posting_payload(
                    layout_rows, ordinary_entry_bytes, tuple_bytes, distinct_keys,
                )
            else:
                leaf_payload += layout_rows * ordinary_entry_bytes
        return leaf_payload, average_entry_bytes

    def _index_bytes(self, leaf_payload, ordinary_entry_bytes):
        """Box 3: pack leaves at fillfactor, then grow internal pages by fanout."""
        usable_page_bytes = (
            (BTREE.PAGE_BYTES - BTREE.PAGE_HEADER_BYTES)
            * BTREE.LEAF_FILLFACTOR / 100.0
        )
        leaf_pages = max(1, int(math.ceil(leaf_payload / usable_page_bytes)))
        total_pages = 1 + leaf_pages
        level_pages = leaf_pages
        internal_usable = (
            (BTREE.PAGE_BYTES - BTREE.PAGE_HEADER_BYTES)
            * BTREE.INTERNAL_FILLFACTOR / 100.0
        )
        fanout = max(2, int(internal_usable // ordinary_entry_bytes))
        while level_pages > 1:
            level_pages = int(math.ceil(level_pages / float(fanout)))
            total_pages += level_pages
        return int(total_pages * BTREE.PAGE_BYTES)

    def statistical_btree_size(self, candidate):
        relation_rows, column_stats = self._relation_statistics(
            candidate.schema, candidate.table,
        )
        if relation_rows <= 0 or any(key not in column_stats for key in candidate.keys):
            return 0
        leaf_payload, average_entry_bytes = self._leaf_payload(
            relation_rows,
            [column_stats[key] for key in candidate.keys],
            candidate.unique,
        )
        return self._index_bytes(leaf_payload, average_entry_bytes)

    def calibrate_candidate_sizes(self, candidates):
        # Existing indexes may contain bloat or have been filled incrementally.
        # Their physical/statistical ratio does not predict a fresh index build.
        for candidate in candidates:
            statistical = self.statistical_btree_size(candidate)
            if statistical <= 0:
                raise RuntimeError(
                    "size estimation requires pg_stats for every key of %s (%s.%s: %s)"
                    % (candidate.id, candidate.schema, candidate.table,
                       ", ".join(candidate.keys))
                )
            candidate.estimated_size = statistical
        return {
            "method": "catalog_statistics_fresh_btree",
            "formula": "postgres_btree_tuple_layout_v3",
            "uses_workload_execution": False,
            "uses_real_latency": False,
            "uses_physical_index_sizes": False,
        }
