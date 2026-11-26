"""Generate a plain-English description of every table with a local LLM (Ollama).

Implements the description extension (implementation_plan.md, §10). Descriptions are written
from the schema only (table and column names, keys and foreign-key links), never from questions,
and for every table of every collection, so test databases get no advantage over distractors.
Calls run with temperature 0 and a fixed seed, pause periodically so the local GPU can cool down,
and results are cached so an interrupted run resumes where it stopped.
"""

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.common import PROJECT_ROOT, load_config, read_json, setup_logging

logger = logging.getLogger(__name__)
PROMPT_VERSION = 1

SYSTEM_PROMPT = """You are a senior database documentation writer. You write short, factual \
descriptions of relational tables for a search index. The index is used to match plain-English \
questions from non-expert users to the right table, so your descriptions must use the everyday \
words such users would say.

Rules:
1. Write 2 or 3 sentences, at most 70 words, as plain prose.
2. Sentence 1: what one row of the table represents, in everyday words; add a common synonym \
when one exists.
3. Sentence 2: the main information the table holds, naming its most important columns in plain \
words (expand abbreviations when their meaning is clear from the names).
4. Sentence 3, only if links are given: what the table connects to and what joining it allows.
5. If the table mainly links two other tables, say that it records the relationship between them.
6. Use only the information given. Do not invent values, numbers, units or columns. If a name is \
unclear, describe it neutrally instead of guessing.
7. Output only the description: no title, no preamble, no lists, no markdown."""

# Few-shot examples on invented schemas (not taken from any benchmark), one per kind of table:
# an entity table, a junction table, a lookup table with abbreviations, a financial table with
# run-together names, and a transaction table with terse column names.
FEW_SHOT: list[tuple[str, str]] = [
    (
        """Database: city_library (in words: city library)
Table: catalogue_item (in words: catalogue item)
Columns: item_id (item id), title (title), author_name (author name), publish_year (publish year), \
genre (genre), branch_id (branch id)
Primary key: item_id
Links: branch_id -> branch.branch_id
Referenced by: loan.item_id""",
        "Each row is a catalogue item (a book or other title) held by the city library. It "
        "records the item's title, author, publication year and genre. It links to the branch "
        "that holds the item and to the loans that record who borrowed it.",
    ),
    (
        """Database: clinic_records (in words: clinic records)
Table: patient_doctor (in words: patient doctor)
Columns: patient_id (patient id), doctor_id (doctor id), first_visit (first visit)
Primary key: patient_id, doctor_id
Links: patient_id -> patient.patient_id; doctor_id -> doctor.doctor_id
Referenced by: (none)""",
        "Each row records that a patient is treated by a doctor, together with the date of their "
        "first visit. It is the relationship table between patients and doctors, so joining it "
        "answers which doctors treat a patient and which patients a doctor sees.",
    ),
    (
        """Database: freight_ops (in words: freight ops)
Table: shp_mode_cd (in words: shp mode cd)
Columns: mode_cd (mode cd), mode_desc (mode desc), max_wt (max wt)
Primary key: mode_cd
Links: (none)
Referenced by: shipment.mode_cd""",
        "Each row is a shipping mode (a transport method) identified by a short code. It gives "
        "the code, a text description of the mode and its maximum weight. Shipments refer to it "
        "to say how they were sent.",
    ),
    (
        """Database: corp_ledger (in words: corp ledger)
Table: LEASEPAYMENTSCHEDULE (in words: LEASE PAYMENT SCHEDULE)
Columns: LEASEPAYMENTSCHEDULEID (LEASE PAYMENT SCHEDULE ID), HASDUEDATE (HAS DUE DATE), \
HASAMOUNT (HAS AMOUNT), ISPAIDBY (IS PAID BY)
Primary key: LEASEPAYMENTSCHEDULEID
Links: ISPAIDBY -> COMPANY.COMPANYID
Referenced by: (none)""",
        "Each row is a scheduled lease payment (an instalment owed under a lease). It records the "
        "payment's due date and amount. It links to the company that pays it, so it shows which "
        "lease payments a company owes and when.",
    ),
    (
        """Database: retail_sales (in words: retail sales)
Table: txn (in words: txn)
Columns: txn_id (txn id), txn_dt (txn dt), cust_id (cust id), store_no (store no), amt (amt), \
pay_type (pay type)
Primary key: txn_id
Links: cust_id -> customer.cust_id; store_no -> store.store_no
Referenced by: txn_item.txn_id""",
        "Each row is a sales transaction (a purchase) made by a customer at a store. It records "
        "the transaction date, the amount paid and the payment type. It links to the customer "
        "and the store, and to the items bought in the transaction.",
    ),
]


def table_prompt(
    db_id: str, table: dict[str, Any], referenced_by: list[str], max_columns: int
) -> str:
    """Format one table's schema in the same layout as the few-shot examples.

    Args:
        db_id: Database identifier.
        table: Table entry from schemas.json.
        referenced_by: "table.column" entries of other tables whose foreign keys point here.
        max_columns: Columns listed in full; the rest are summarised as a count.

    Returns:
        The user message for this table.
    """
    columns = table["columns"]
    shown = ", ".join(f"{c['name']} ({c['normalized_name']})" for c in columns[:max_columns])
    if len(columns) > max_columns:
        shown += f", and {len(columns) - max_columns} more columns"
    keys = [c["name"] for c in columns if c.get("primary_key")]
    links = [
        f"{c['name']} -> {c['foreign_key']['table']}.{c['foreign_key']['column']}"
        for c in columns
        if c.get("foreign_key")
    ]
    return "\n".join(
        [
            f"Database: {db_id} (in words: {db_id.replace('_', ' ')})",
            f"Table: {table['name']} (in words: {table['normalized_name']})",
            f"Columns: {shown}",
            f"Primary key: {', '.join(keys) or '(none)'}",
            f"Links: {'; '.join(links) or '(none)'}",
            f"Referenced by: {'; '.join(referenced_by) or '(none)'}",
        ]
    )


def referenced_by_index(tables: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each table name to the "table.column" foreign keys of other tables that point to it."""
    index: dict[str, list[str]] = {}
    for table in tables:
        for column in table["columns"]:
            foreign_key = column.get("foreign_key")
            if foreign_key and foreign_key["table"] != table["name"]:
                index.setdefault(foreign_key["table"], []).append(
                    f"{table['name']}.{column['name']}"
                )
    return index


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Return the chat messages: system role, the few-shot pairs, then the table to describe."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_input, example_output in FEW_SHOT:
        messages += [
            {"role": "user", "content": example_input},
            {"role": "assistant", "content": example_output},
        ]
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_ollama(messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    """Send one chat request to the local Ollama server and return the cleaned reply.

    Args:
        messages: Chat messages.
        settings: Model settings (url, model, temperature, seed, max_tokens, think).

    Returns:
        The reply text on one line, without a leading "Description:" label.
    """
    payload = {
        "model": settings["model"],
        "messages": messages,
        "stream": False,
        "think": settings.get("think", False),
        "options": {
            "temperature": settings["temperature"],
            "seed": settings["seed"],
            "num_predict": settings["max_tokens"],
        },
    }
    request = urllib.request.Request(
        settings["url"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=settings["timeout"]) as response:
        reply = json.loads(response.read())["message"]["content"]
    reply = " ".join(reply.split())
    return reply.removeprefix("Description:").strip()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load already generated descriptions (one JSON object per line), keyed by table id."""
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["table_id"]: row for row in map(json.loads, handle) if row.get("description")}


def describe_collection(collection: str, config: dict[str, Any], limit: int | None) -> int:
    """Describe every table of one collection, appending new descriptions to its cache file.

    Args:
        collection: "spider", "bird" or "fiben".
        config: Description configuration (model, cooldown, paths).
        limit: Stop after this many new descriptions (None: no limit).

    Returns:
        Number of new descriptions written.
    """
    schemas = read_json(PROJECT_ROOT / "data" / "processed" / collection / "schemas.json")
    path = PROJECT_ROOT / config["output_dir"] / f"{collection}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(path)
    written = 0
    cooldown = config["cooldown"]
    with path.open("a") as handle:
        for db_id, tables in schemas.items():
            referenced = referenced_by_index(tables)
            for table in tables:
                table_id = f"{db_id}.{table['name']}"
                if table_id in done:
                    continue
                if limit is not None and written >= limit:
                    return written
                prompt = table_prompt(
                    db_id, table, referenced.get(table["name"], []), config["max_columns"]
                )
                started = time.perf_counter()
                description = call_ollama(build_messages(prompt), config["llm"])
                row = {"table_id": table_id, "description": description,
                       "model": config["llm"]["model"], "prompt_version": PROMPT_VERSION,
                       "seconds": round(time.perf_counter() - started, 2)}  # fmt: skip
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                if written % config["log_every"] == 0:
                    logger.info("%s: %d new descriptions (%s)", collection, written, table_id)
                # Pause regularly so the local GPU does not run at full load for an hour.
                if written % cooldown["every_calls"] == 0:
                    time.sleep(cooldown["seconds"])
    return written


def main() -> None:
    """Describe the tables of the requested collections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/descriptions.yaml")
    parser.add_argument("--collections", nargs="+", default=["spider", "bird", "fiben"])
    parser.add_argument("--limit", type=int, default=None, help="stop after N new descriptions")
    parser.add_argument("--model", default=None, help="override the configured model")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.model:
        config["llm"]["model"] = args.model
    remaining = args.limit
    for collection in args.collections:
        written = describe_collection(collection, config, remaining)
        logger.info("%s: wrote %d new descriptions", collection, written)
        if remaining is not None:
            remaining -= written
            if remaining <= 0:
                break


if __name__ == "__main__":
    main()
