"""Stage 1: extract workload features and write the budget for the storage proportion."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis, Where
from sqlparse.tokens import Keyword

from core.common import half_cpu_workers


QID_MARKER = re.compile(
    r"^--\s*===\s*qid:\s*(?P<qid>\S+?)(?:\s+.*?)?\s*===\s*$", re.MULTILINE
)
SUMMARY_PATTERN_LIMIT = 8
LINE_COMMENT = re.compile(r"--[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

JOIN_EQUALITY = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
)
BARE_JOIN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\b")
GROUP_BY = re.compile(
    r"\bGROUP\s+BY\s+(?P<body>.*?)(?=\b(?:ORDER\s+BY|HAVING|LIMIT|UNION|WINDOW)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
ORDER_BY = re.compile(
    r"\bORDER\s+BY\s+(?P<body>.*?)(?=\b(?:LIMIT|UNION|WINDOW)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
QUALIFIED_COLUMN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

STRING_LIT = re.compile(r"'(?:[^']|'')*'")
DATE_KW = re.compile(r"\bdate\s+\?", re.IGNORECASE)
INTERVAL_KW = re.compile(r"\binterval\s+\?(?:\s+[A-Za-z]+)?", re.IGNORECASE)
NUMBER_LIT = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?(?![A-Za-z_])")

SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "order",
    "by",
    "having",
    "limit",
    "and",
    "or",
    "not",
    "in",
    "like",
    "ilike",
    "between",
    "is",
    "null",
    "as",
    "on",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "full",
    "cross",
    "union",
    "all",
    "distinct",
    "case",
    "when",
    "then",
    "else",
    "end",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "cast",
    "extract",
    "interval",
    "date",
    "exists",
    "with",
    "asc",
    "desc",
    "over",
    "partition",
    "substring",
    "coalesce",
    "true",
    "false",
    "create",
    "view",
    "drop",
    "insert",
    "update",
    "delete",
    "values",
    "set",
    "into",
    "table",
}

# prefix hint used when catalog is absent (SSB / TPC-H / TPC-DS / DSB / JOB)
PREFIX_HINT = {
    "l": "lineitem",
    "o": "orders",
    "c": "customer",
    "p": "part",
    "s": "supplier",
    "n": "nation",
    "r": "region",
    "ps": "partsupp",
    "lo": "lineorder",
    "ss": "store_sales",
    "sr": "store_returns",
    "cs": "catalog_sales",
    "cr": "catalog_returns",
    "ws": "web_sales",
    "wr": "web_returns",
    "ca": "customer_address",
    "cd": "customer_demographics",
    "hd": "household_demographics",
    "i": "item",
    "t": "title",
    "w": "warehouse",
    "cc": "call_center",
    "cp": "catalog_page",
    "wp": "web_page",
    "sm": "ship_mode",
    "ib": "income_band",
    "inv": "inventory",
    "mc": "movie_companies",
    "mi": "movie_info",
    "ci": "cast_info",
    "cn": "company_name",
    "ct": "company_type",
    "it": "info_type",
    "kt": "kind_type",
    "lt": "link_type",
    "rt": "role_type",
    "an": "aka_name",
}

PRED_PATTERNS = [
    (
        "isnull",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+IS\s+(?:NOT\s+)?NULL",
            re.IGNORECASE,
        ),
    ),
    (
        "in",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+(?:NOT\s+)?IN\s*\(",
            re.IGNORECASE,
        ),
    ),
    (
        "like",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+(?:NOT\s+)?I?LIKE\b",
            re.IGNORECASE,
        ),
    ),
    (
        "between",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+BETWEEN\b",
            re.IGNORECASE,
        ),
    ),
    (
        "range",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*(?:<=|>=|<|>|<>|!=)\s*"
        ),
    ),
    (
        "eq",
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*=\s*(?P<value>'[^']*'|-?\d+(?:\.\d+)?|\?)"
        ),
    ),
]


class ColumnRef:
    def __init__(self, table, column):
        self.table = table
        self.column = column

    def key(self):
        return self.table + "." + self.column

    def __str__(self):
        return self.key()

    def __eq__(self, other):
        return isinstance(other, ColumnRef) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class Catalog:
    def __init__(self):
        self.tables = {}
        self.existing = []
        self.ownership = {}
        self.db_size = 0
        self.ok = False

    def connect(self, host, port, user, password, dbname):
        if not host or not dbname:
            print("catalog: skipped (no host/dbname)")
            return False
        try:
            import psycopg2
        except ImportError:
            print("catalog: psycopg2 not installed, skip")
            return False
        try:
            kwargs = dict(host=host, port=int(port), user=user, dbname=dbname)
            if password:
                kwargs["password"] = password
            conn = psycopg2.connect(**kwargs)
            conn.autocommit = True
            cur = conn.cursor()
            self._load_stats(cur)
            self._load_existing(cur)
            cur.execute("SELECT pg_database_size(current_database())")
            self.db_size = int(cur.fetchone()[0] or 0)
            cur.close()
            conn.close()
            self.ok = True
            print(
                "catalog: loaded",
                len(self.tables),
                "tables,",
                len(self.existing),
                "indexes",
            )
            return True
        except Exception as e:
            print("catalog: connect failed,", e)
            return False

    def _load_stats(self, cur):
        cur.execute(
            """
            SELECT c.relname, COALESCE(c.reltuples, 0), pg_table_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r'
            """
        )
        for name, rows, size in cur.fetchall():
            self.tables[name] = {
                "rows": float(rows or 0),
                "bytes": int(size or 0),
                "columns": {},
            }

        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        )
        for table, column, data_type in cur.fetchall():
            if table not in self.tables:
                continue
            self.tables[table]["columns"][column] = {
                "ndv": None,
                "top_value_fraction": None,
                "data_type": data_type,
            }
            self.ownership.setdefault(column.lower(), []).append(table)

        cur.execute(
            """
            SELECT tablename, attname, n_distinct, most_common_freqs
            FROM pg_stats WHERE schemaname = 'public'
            """
        )
        for table, column, n_distinct, mcf in cur.fetchall():
            if table not in self.tables or column not in self.tables[table]["columns"]:
                continue
            rows = self.tables[table]["rows"]
            if n_distinct is None:
                ndv = None
            elif float(n_distinct) >= 0:
                ndv = float(n_distinct)
            else:
                ndv = -float(n_distinct) * rows if rows else None
            topf = None
            if mcf:
                try:
                    topf = float(mcf[0])
                except Exception:
                    topf = None
            self.tables[table]["columns"][column]["ndv"] = ndv
            self.tables[table]["columns"][column]["top_value_fraction"] = topf

    def _load_existing(self, cur):
        cur.execute(
            """
            SELECT c.relname, ic.relname, i.indnkeyatts, i.indisprimary,
                   pg_get_indexdef(i.indexrelid), pg_relation_size(i.indexrelid)
            FROM pg_index i
            JOIN pg_class ic ON ic.oid = i.indexrelid
            JOIN pg_class c ON c.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
            """
        )
        for table, index_name, nkey, is_pk, indexdef, size in cur.fetchall():
            keys = parse_indexdef_keys(indexdef, nkey)
            if not keys:
                continue
            self.existing.append(
                {
                    "name": index_name,
                    "table": table,
                    "keys": keys,
                    "size": int(size or 0),
                    "pk": bool(is_pk),
                }
            )

    def column_stats(self, table, column):
        tb = self.tables.get(table)
        if not tb:
            return None
        return tb["columns"].get(column)


def parse_indexdef_keys(indexdef, nkey):
    match = re.search(r"\((?P<keys>[^()]*)\)", indexdef or "")
    if not match:
        return []
    keys = []
    for part in match.group("keys").split(","):
        tok = part.strip().split()[0].lower() if part.strip() else ""
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", tok):
            return []
        keys.append(tok)
    if nkey:
        keys = keys[: int(nkey)]
    return keys


def strip_comments(sql):
    return LINE_COMMENT.sub(" ", BLOCK_COMMENT.sub(" ", sql))


def compact_num(value):
    if value is None:
        return None
    value = float(value)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return "%.1f%s" % (value / scale, unit)
    return "%.0f" % value


def compact_bytes(value):
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return "%.1f%s" % (size, unit)
        size /= 1024
    return "%.1fTB" % size


def storage_constraint(db_size_bytes, proportion):
    db_mb = float(db_size_bytes or 0) / (1024.0 * 1024.0)
    limit_mb = db_mb * float(proportion)
    return {
        "database_size": compact_bytes(db_size_bytes),
        "database_size_mb": round(db_mb, 1),
        "index_storage_proportion": float(proportion),
        "storage_constraint": compact_bytes(limit_mb * 1024 * 1024),
        "storage_constraint_mb": int(limit_mb),
    }

class _ParseMixin:
    def load_workload(self, workload_path):
        text = open(workload_path, "r", errors="ignore").read()
        matches = list(QID_MARKER.finditer(text))
        blocks = []
        if matches:
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                blocks.append((match.group("qid"), text[match.end() : end]))
        else:
            blocks.append((os.path.splitext(os.path.basename(workload_path))[0], text))

        def _parse_block(item):
            qid, body = item
            sql = self._pick_sql(body)
            if not sql:
                return None
            feat = self.extract_expressions(sql)
            feat["qid"] = qid
            feat["sql"] = sql
            feat["template"] = self.sql2template(sql)
            return feat

        workers = min(half_cpu_workers(), max(1, len(blocks)))
        if workers > 1 and len(blocks) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parsed = list(pool.map(_parse_block, blocks))
        else:
            parsed = [_parse_block(item) for item in blocks]
        self.queries = [feat for feat in parsed if feat]
        print("loaded queries :", len(self.queries))

    def _pick_sql(self, body):
        cleaned = strip_comments(body)
        parts = [p.strip().rstrip(";") for p in re.split(r";\s*", cleaned) if p.strip()]
        for pos, stmt in enumerate(parts):
            view = re.match(
                r"^CREATE\s+VIEW\s+(\w+)\s*(\([^)]*\))?\s+AS\s+(.*)$",
                stmt,
                re.IGNORECASE | re.DOTALL,
            )
            if not view:
                continue
            outer = next(
                (
                    item
                    for item in parts[pos + 1 :]
                    if item.lstrip().upper().startswith(("SELECT", "WITH"))
                ),
                None,
            )
            if outer:
                columns = view.group(2) or ""
                return "WITH %s%s AS (%s) %s" % (
                    view.group(1),
                    columns,
                    view.group(3),
                    outer,
                )
        mains = []
        for stmt in parts:
            head = stmt.lstrip().upper()
            if head.startswith("SELECT") or head.startswith("WITH"):
                mains.append(stmt)
            elif head.startswith("CREATE") and re.search(r"\bSELECT\b", stmt, re.I):
                mains.append(stmt)
        if not mains:
            return None
        return " ".join(mains)

    # ---- SQL2Template ---------------------------------------------------- #
    def sql2template(self, sql):
        text = " ".join(strip_comments(sql).split())
        text = STRING_LIT.sub("?", text)
        text = DATE_KW.sub("?", text)
        text = INTERVAL_KW.sub("?", text)
        text = NUMBER_LIT.sub("?", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---- expression extraction ------------------------------------------- #
    def extract_expressions(self, sql):
        text = " ".join(strip_comments(sql).split())
        derived = self._derived_names(text)
        aliases = self._extract_aliases(self._mask_func_from(text))
        used_tables = sorted(t for t in set(aliases.values()) if t not in derived)
        aliases = {
            k: v for k, v in aliases.items() if v not in derived and k not in derived
        }

        joins = []
        join_cols = set()
        for a_alias, a_col, b_alias, b_col in JOIN_EQUALITY.findall(text):
            left = self._resolve(a_alias, a_col, aliases, used_tables)
            right = self._resolve(b_alias, b_col, aliases, used_tables)
            if left is None or right is None or left.table == right.table:
                continue
            pair = [left.table, left.column, right.table, right.column]
            joins.append(pair)
            join_cols.add(left.key())
            join_cols.add(right.key())

        for left_tok, right_tok in BARE_JOIN.findall(text):
            if "." in left_tok or "." in right_tok:
                continue
            if left_tok.lower() in SQL_KEYWORDS or right_tok.lower() in SQL_KEYWORDS:
                continue
            left = self._resolve_bare(left_tok, used_tables)
            right = self._resolve_bare(right_tok, used_tables)
            if left is None or right is None or left.table == right.table:
                continue
            pair = [left.table, left.column, right.table, right.column]
            if (
                pair in joins
                or [right.table, right.column, left.table, left.column] in joins
            ):
                continue
            joins.append(pair)
            join_cols.add(left.key())
            join_cols.add(right.key())

        filters = []
        seen_filter = set()
        for kind, pattern in PRED_PATTERNS:
            for match in pattern.finditer(text):
                # The last operand of an expression is not a scalar predicate:
                # price / list_price BETWEEN ... must not become list_price BETWEEN.
                if re.search(r"(?:[+*/%|-]|::)\s*$", text[:match.start(1)]):
                    continue
                ref = self._resolve_token(match.group(1), aliases, used_tables)
                if ref is None or ref.key() in join_cols:
                    continue
                mark = (ref.key(), kind)
                if mark in seen_filter:
                    continue
                seen_filter.add(mark)
                filters.append(
                    {
                        "table": ref.table,
                        "column": ref.column,
                        "op": kind,
                    }
                )

        groups = self._clause_columns(GROUP_BY, text, aliases, used_tables)
        orders = self._order_columns(text, aliases, used_tables)

        return {
            "tables": used_tables,
            "aliases": aliases,
            "filters": filters,
            "joins": joins,
            "groups": groups,
            "orders": orders,
        }

    def _mask_func_from(self, text):
        text = re.sub(r"\bEXTRACT\s*\([^)]*\)", " ", text, flags=re.I)
        text = re.sub(r"\bSUBSTR(?:ING)?\s*\([^)]*\)", " ", text, flags=re.I)
        return text

    def _derived_names(self, text):
        names = set()
        for name in re.findall(r"\bCREATE\s+VIEW\s+(\w+)", text, re.I):
            names.add(name.lower())
        for name in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s+AS\s*\(",
            text,
            re.I,
        ):
            names.add(name.lower())
        return names

    def _extract_aliases(self, text):
        # Walk nested FROM/JOIN groups rather than letting an outer regex consume
        # a subquery's FROM, or lose comma joins following an ON expression.
        aliases = {}

        def remember(token):
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    remember(identifier)
            elif isinstance(token, Identifier):
                if any(isinstance(child, Parenthesis) for child in token.tokens):
                    return
                table = token.get_real_name()
                if table:
                    table = table.lower()
                    aliases[(token.get_alias() or table).lower()] = table
                    aliases.setdefault(table, table)

        def walk(group):
            in_from = expect_relation = False
            for token in group.tokens:
                if token.is_whitespace:
                    continue
                if token.is_group:
                    walk(token)
                if isinstance(token, Where):
                    in_from = expect_relation = False
                elif token.ttype in Keyword:
                    word = token.normalized
                    if word == "FROM" or word.endswith("JOIN"):
                        in_from = expect_relation = True
                    elif word in {"WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
                                  "UNION", "UNION ALL", "EXCEPT", "INTERSECT", "WINDOW"}:
                        in_from = expect_relation = False
                    elif word == "ON":
                        expect_relation = False
                elif in_from and (expect_relation or isinstance(token, IdentifierList)):
                    remember(token)
                    expect_relation = False
                elif in_from and token.value == ",":
                    expect_relation = True

        for statement in sqlparse.parse(text):
            walk(statement)
        return aliases

    def _known_column(self, table, column):
        if not self.catalog.ok:
            return True
        return self.catalog.column_stats(table, column) is not None

    def _ref(self, table, column):
        if not table or not column or column.lower() in SQL_KEYWORDS:
            return None
        if not self._known_column(table, column.lower()):
            return None
        return ColumnRef(table, column.lower())

    def _resolve(self, alias, column, aliases, used_tables):
        if column.lower() in SQL_KEYWORDS:
            return None
        table = aliases.get(alias.lower())
        if table is None:
            return self._resolve_bare(column, used_tables)
        return self._ref(table, column)

    def _resolve_token(self, token, aliases, used_tables):
        token = token.strip()
        if "." in token:
            alias, column = token.split(".", 1)
            return self._resolve(alias, column, aliases, used_tables)
        return self._resolve_bare(token, used_tables)

    def _resolve_bare(self, name, used_tables):
        name = name.lower()
        if name in SQL_KEYWORDS:
            return None
        owners = [t for t in self.catalog.ownership.get(name, []) if t in used_tables]
        if len(owners) == 1:
            return self._ref(owners[0], name)
        prefix = name.split("_", 1)[0]
        pool = owners if owners else list(used_tables)
        for table in pool:
            if table.startswith(prefix) or prefix.startswith(table[:2]):
                ref = self._ref(table, name)
                if ref is not None:
                    return ref
        if prefix in PREFIX_HINT and PREFIX_HINT[prefix] in used_tables:
            ref = self._ref(PREFIX_HINT[prefix], name)
            if ref is not None:
                return ref
        if prefix == "d":
            for table in ("dates", "date_dim", "date"):
                if table in used_tables:
                    ref = self._ref(table, name)
                    if ref is not None:
                        return ref
        if prefix == "s":
            for table in ("supplier", "store"):
                if table in used_tables:
                    ref = self._ref(table, name)
                    if ref is not None:
                        return ref
        if prefix == "c":
            for table in ("customer", "company_name"):
                if table in used_tables:
                    ref = self._ref(table, name)
                    if ref is not None:
                        return ref
        if len(owners) > 1:
            return self._ref(sorted(owners)[0], name)
        if len(used_tables) == 1:
            return self._ref(used_tables[0], name)
        return None

    def _clause_columns(self, pattern, text, aliases, used_tables):
        match = pattern.search(text)
        if not match:
            return []
        out = []
        seen = set()
        for alias, column in QUALIFIED_COLUMN.findall(match.group("body")):
            ref = self._resolve(alias, column, aliases, used_tables)
            if ref is None or ref.key() in seen:
                continue
            seen.add(ref.key())
            out.append({"table": ref.table, "column": ref.column})
        for token in re.split(r",", match.group("body")):
            tok = token.strip().split()[0] if token.strip() else ""
            tok = re.sub(r"[()]", "", tok)
            if not tok or "." in tok or tok.lower() in SQL_KEYWORDS:
                continue
            ref = self._resolve_bare(tok, used_tables)
            if ref is None or ref.key() in seen:
                continue
            seen.add(ref.key())
            out.append({"table": ref.table, "column": ref.column})
        return out

    def _order_columns(self, text, aliases, used_tables):
        match = ORDER_BY.search(text)
        if not match:
            return []
        out = []
        seen = set()
        for item in match.group("body").split(","):
            item = item.strip()
            if not item:
                continue
            direction = "ASC"
            if re.search(r"\bDESC\b", item, re.I):
                direction = "DESC"
            q = QUALIFIED_COLUMN.search(item)
            if q:
                ref = self._resolve(q.group(1), q.group(2), aliases, used_tables)
            else:
                tok = re.sub(r"\b(?:ASC|DESC)\b", "", item, flags=re.I).strip()
                tok = re.split(r"[\s()]+", tok)[0]
                ref = self._resolve_bare(tok, used_tables)
            if ref is None or ref.key() in seen:
                continue
            seen.add(ref.key())
            out.append(
                {
                    "table": ref.table,
                    "column": ref.column,
                    "direction": direction,
                }
            )
        return out

class _FeatureMixin:
    def parse_workload(self, workload_path):
        self.load_workload(workload_path)
        self._aggregate()
        self._build_patterns()
        self.existing = list(self.catalog.existing)
        print(
            "tables :",
            len(self.tables),
            "columns :",
            len(self.columns),
            "patterns :",
            len(self.patterns),
        )

    def _col_entry(self, table, column):
        if not self._known_column(table, column):
            return None
        key = table + "." + column
        if key not in self.columns:
            stats = self.catalog.column_stats(table, column) or {}
            self.columns[key] = {
                "table": table,
                "column": column,
                "eq_freq": 0,
                "range_freq": 0,
                "join_freq": 0,
                "join_peer": None,
                "join_peers": {},
                "order_freq": 0,
                "order_direction": None,
                "order_dirs": {},
                "ndv": stats.get("ndv"),
                "top_value_fraction": stats.get("top_value_fraction"),
            }
        if table not in self.tables:
            tb = self.catalog.tables.get(table, {})
            self.tables[table] = {
                "rows": tb.get("rows"),
            }
        return self.columns[key]

    def _role_freq(self, entry):
        return (
            entry["eq_freq"]
            + entry["range_freq"]
            + entry["join_freq"]
            + entry["order_freq"]
        )

    def _aggregate(self):
        self.tables = {}
        self.columns = {}
        for q in self.queries:
            for table in q["tables"]:
                if table not in self.tables:
                    tb = self.catalog.tables.get(table, {})
                    self.tables[table] = {
                        "rows": tb.get("rows"),
                    }
            for pred in q["filters"]:
                entry = self._col_entry(pred["table"], pred["column"])
                if entry is None:
                    continue
                if pred["op"] in ("eq", "in", "isnull"):
                    entry["eq_freq"] += 1
                else:
                    entry["range_freq"] += 1
            for a_tb, a_col, b_tb, b_col in q["joins"]:
                left = self._col_entry(a_tb, a_col)
                right = self._col_entry(b_tb, b_col)
                if left is None or right is None:
                    continue
                left["join_freq"] += 1
                right["join_freq"] += 1
                left["join_peers"][b_tb + "." + b_col] = (
                    left["join_peers"].get(b_tb + "." + b_col, 0) + 1
                )
                right["join_peers"][a_tb + "." + a_col] = (
                    right["join_peers"].get(a_tb + "." + a_col, 0) + 1
                )
            for item in q["orders"]:
                entry = self._col_entry(item["table"], item["column"])
                if entry is None:
                    continue
                entry["order_freq"] += 1
                d = item.get("direction") or "ASC"
                entry["order_dirs"][d] = entry["order_dirs"].get(d, 0) + 1

        self.columns = {k: v for k, v in self.columns.items() if self._role_freq(v) > 0}

        for entry in self.columns.values():
            if entry["order_dirs"]:
                entry["order_direction"] = max(
                    entry["order_dirs"].items(), key=lambda x: x[1]
                )[0]

        ranked = sorted(
            self.columns.values(),
            key=lambda e: (
                -self._role_freq(e),
                e["table"],
                e["column"],
            ),
        )
        self.column_ids = {}
        self.id_of = {}
        for i, entry in enumerate(ranked, 1):
            cid = "C%d" % i
            key = entry["table"] + "." + entry["column"]
            self.column_ids[cid] = key
            self.id_of[key] = cid
            entry["cid"] = cid
        for entry in self.columns.values():
            if entry["join_peers"]:
                peer = max(entry["join_peers"].items(), key=lambda x: x[1])[0]
                entry["join_peer"] = self.id_of.get(peer)

    def _qid_of(self, table, column):
        return self.id_of.get(table + "." + column)

    def _build_patterns(self):
        groups = {}
        for q in self.queries:
            eq = []
            rng = []
            seen_eq = set()
            seen_rng = set()
            for pred in q["filters"]:
                cid = self._qid_of(pred["table"], pred["column"])
                if cid is None:
                    continue
                if pred["op"] in ("eq", "in", "isnull"):
                    if cid not in seen_eq:
                        eq.append(cid)
                        seen_eq.add(cid)
                else:
                    if cid not in seen_rng:
                        rng.append(cid)
                        seen_rng.add(cid)
            orders = []
            seen_ord = set()
            for item in q["orders"]:
                cid = self._qid_of(item["table"], item["column"])
                if cid is None or cid in seen_ord:
                    continue
                seen_ord.add(cid)
                orders.append((cid, item.get("direction") or "ASC"))
            joins = []
            seen_join = set()
            for a_tb, a_col, b_tb, b_col in q["joins"]:
                left = self._qid_of(a_tb, a_col)
                right = self._qid_of(b_tb, b_col)
                if left is None or right is None:
                    continue
                pair = tuple(sorted((left, right), key=lambda x: int(x[1:])))
                if pair in seen_join:
                    continue
                seen_join.add(pair)
                joins.append(pair)
            eq_key = tuple(sorted(eq, key=lambda x: int(x[1:])))
            rng_key = tuple(rng)
            ord_key = tuple(orders)
            join_key = tuple(
                sorted(joins, key=lambda x: (int(x[0][1:]), int(x[1][1:])))
            )
            if not (eq_key or rng_key or ord_key or join_key):
                continue
            sig = (eq_key, rng_key, ord_key, join_key)
            if sig not in groups:
                groups[sig] = {
                    "eq": list(eq_key),
                    "range": list(rng_key),
                    "order": list(ord_key),
                    "joins": list(join_key),
                    "freq": 0,
                    "tables": set(),
                    "qids": [],
                }
            groups[sig]["freq"] += 1
            groups[sig]["tables"].update(q["tables"])
            groups[sig]["qids"].append(q["qid"])

        items = list(groups.values())
        items.sort(
            key=lambda g: (
                -g["freq"],
                -(len(g["eq"]) + len(g["range"]) + len(g["joins"]) + len(g["order"])),
            )
        )
        for g in items:
            g["tables"] = sorted(g["tables"])
        self.patterns = items

    def pattern_text(self, g):
        parts = []
        if g["eq"]:
            parts.append("EQ{" + ",".join(g["eq"]) + "}")
        if g["joins"]:
            js = ",".join("%s=%s" % (cid, peer) for cid, peer in g["joins"])
            parts.append("JOIN(" + js + ")")
        if g["range"]:
            parts.append("RANGE(" + ",".join(g["range"]) + ")")
        if g["order"]:
            bits = []
            for cid, direction in g["order"]:
                bits.append(cid + (" " + direction if direction != "ASC" else ""))
            parts.append("ORDER(" + ", ".join(bits) + ")")
        return " + ".join(parts)

    def _kept_query(self, q):
        filters = [
            pred for pred in q["filters"] if self._qid_of(pred["table"], pred["column"])
        ]
        joins = [
            pair
            for pair in q["joins"]
            if self._qid_of(pair[0], pair[1]) and self._qid_of(pair[2], pair[3])
        ]
        groups = [
            item for item in q["groups"] if self._qid_of(item["table"], item["column"])
        ]
        orders = [
            item for item in q["orders"] if self._qid_of(item["table"], item["column"])
        ]
        out = {
            "qid": q["qid"],
            "template": q["template"],
            "tables": q["tables"],
        }
        if filters:
            out["filters"] = filters
        if joins:
            out["joins"] = joins
        if groups:
            out["groups"] = groups
        if orders:
            out["orders"] = orders
        return out

    # ---- render / dump --------------------------------------------------- #
    def to_dict(self):
        columns = {}
        for key, entry in self.columns.items():
            item = {
                "cid": entry["cid"],
                "table": entry["table"],
                "column": entry["column"],
            }
            if entry["eq_freq"]:
                item["eq_freq"] = entry["eq_freq"]
            if entry["range_freq"]:
                item["range_freq"] = entry["range_freq"]
            if entry["join_freq"]:
                item["join_freq"] = entry["join_freq"]
            if entry.get("join_peer"):
                item["join_peer"] = entry["join_peer"]
            if entry["order_freq"]:
                item["order_freq"] = entry["order_freq"]
            if entry.get("order_direction"):
                item["order_direction"] = entry["order_direction"]
            if entry.get("ndv") is not None:
                item["ndv"] = entry["ndv"]
            topf = entry.get("top_value_fraction")
            if topf is not None and topf >= 0.1:
                item["top_value_fraction"] = topf
            columns[key] = item
        patterns = []
        for g in self.patterns:
            item = {
                "text": self.pattern_text(g),
                "freq": g["freq"],
                "tables": g["tables"],
            }
            if g["eq"]:
                item["eq"] = g["eq"]
            if g["range"]:
                item["range"] = g["range"]
            if g["order"]:
                item["order"] = g["order"]
            if g["joins"]:
                item["joins"] = g["joins"]
            if g["qids"]:
                item["qids"] = g["qids"]
            patterns.append(item)
        existing = []
        for ix in self.existing:
            item = {
                "table": ix["table"],
                "keys": ix["keys"],
                "size": ix["size"],
            }
            if ix.get("pk"):
                item["pk"] = True
            existing.append(item)
        return {
            "queries": len(self.queries),
            "tables": self.tables,
            "columns": columns,
            "column_ids": self.column_ids,
            "patterns": patterns,
            "existing_indexes": existing,
            "storage": self.storage,
            "query_features": [self._kept_query(q) for q in self.queries],
        }

    def render_summary(self):
        lines = []
        lines.append("WORKLOAD SUMMARY")
        lines.append("  queries = %d" % len(self.queries))
        lines.append("")
        for table in sorted(self.tables):
            info = self.tables[table]
            lines.append("TABLE %s" % table)
            if info.get("rows") is not None:
                lines.append("  rows = %s" % compact_num(info["rows"]))
            lines.append("")
        ranked = sorted(self.columns.values(), key=lambda e: int(e["cid"][1:]))
        for entry in ranked:
            lines.append(
                "COLUMN %s = %s.%s" % (entry["cid"], entry["table"], entry["column"])
            )
            if entry["eq_freq"]:
                lines.append("  EQ.freq = %d" % entry["eq_freq"])
            if entry["range_freq"]:
                lines.append("  RANGE.freq = %d" % entry["range_freq"])
            if entry["join_freq"]:
                lines.append("  JOIN.freq = %d" % entry["join_freq"])
            if entry.get("join_peer"):
                lines.append("  JOIN.peer = %s" % entry["join_peer"])
            if entry["order_freq"]:
                lines.append("  ORDER.freq = %d" % entry["order_freq"])
            if entry.get("order_direction"):
                lines.append("  ORDER.direction = %s" % entry["order_direction"])
            if entry.get("ndv") is not None:
                lines.append("  NDV = %s" % compact_num(entry["ndv"]))
            topf = entry.get("top_value_fraction")
            if topf is not None and topf >= 0.1:
                lines.append("  top_value_fraction = %.2f" % topf)
            lines.append("")
        lines.append("ACCESS PATTERNS")
        top = self.patterns[:SUMMARY_PATTERN_LIMIT]
        rest = self.patterns[SUMMARY_PATTERN_LIMIT:]
        for i, g in enumerate(top, 1):
            lines.append("  P%d: %s" % (i, self.pattern_text(g)))
            lines.append("      freq = %d" % g["freq"])
            lines.append("      tables = [%s]" % ", ".join(g["tables"]))
            lines.append("")
        if rest:
            freq = sum(g["freq"] for g in rest)
            lines.append("  others:")
            lines.append("      freq = %d" % freq)
            lines.append("")
        lines.append("EXISTING INDEXES")
        if not self.existing:
            lines.append("  (none)")
            lines.append("")
        else:
            for i, ix in enumerate(self.existing, 1):
                cids = []
                for col in ix["keys"]:
                    cids.append(
                        self.id_of.get(ix["table"] + "." + col, ix["table"] + "." + col)
                    )
                tag = " pk" if ix.get("pk") else ""
                lines.append("  index%d: (%s)%s" % (i, ",".join(cids), tag))
                if ix.get("size"):
                    lines.append("       size = %s" % compact_bytes(ix["size"]))
                lines.append("")
        lines.append("BUDGET")
        if self.storage:
            lines.append("  database_size = %s" % self.storage["database_size"])
            lines.append(
                "  index_storage_proportion = %s"
                % self.storage["index_storage_proportion"]
            )
            lines.append(
                "  storage_constraint = %s" % self.storage["storage_constraint"]
            )
        else:
            lines.append("  storage_constraint = (unknown)")
        lines.append("")
        lines.append("CANDIDATE INDEXES")
        lines.append("  ...")
        lines.append("")
        return "\n".join(lines)

class WorkloadExtractor(_FeatureMixin, _ParseMixin):
    def __init__(self) -> None:
        self.catalog = Catalog()
        self.queries = []
        self.tables = {}
        self.columns = {}
        self.column_ids = {}
        self.id_of = {}
        self.patterns = []
        self.existing = []
        self.storage = None

    def apply_storage(self, proportion):
        self.storage = storage_constraint(self.catalog.db_size, proportion)

def extract_and_save(
    workload_file,
    output_file,
    summary_file,
    host,
    port,
    user,
    password,
    dbname,
    proportion=0.5,
):
    wp = WorkloadExtractor()
    wp.catalog.connect(host, port, user, password, dbname)
    wp.apply_storage(proportion)
    wp.parse_workload(workload_file)
    features = wp.to_dict()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print("features ->", output_file)
    summary = wp.render_summary()
    os.makedirs(os.path.dirname(summary_file) or ".", exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print("summary ->", summary_file)
    return features, summary, wp.storage

