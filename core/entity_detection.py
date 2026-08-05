"""Recall-biased entity detection for multi-entity Sentiment Coding.

Two ways to define entities, mirroring two existing Quadrant scripts:
  - Manual aliases: compiled to a left-\\b-anchored, open-right regex per
    entity (children_safety_classifier_v2.py's _compile_patterns /
    _extract_entities approach) — the left boundary prevents mid-word
    matches, the open right boundary still catches derivative forms
    (YouTubers, TikTokkers). Recall over precision by design; the LLM sorts
    out false positives (e.g. "discord" meaning disagreement) downstream.
  - Brandwatch parent category: parses the "Category Details" column BW
    itself populates when a query is set up with categorized entities
    (stage1_parse.py's extract_entities) — no regex needed, BW already
    tagged the entity; just extract the ones filed under the given parent.
"""
import re
from collections import Counter

# Same two "Category Details" formats stage1_parse.py handles:
#   Format A: {id=N, name=X, parentName=P, parentId=N}
#   Format B: {id=N, name=X, displayName=null, parentId=N, parentName=P}
_CATEGORY_PATTERN_A_TEMPLATE = r"\{id=\d+,\s*name=([^,}]+?),\s*parentName=__PARENT__,\s*parentId=\d+\}"
_CATEGORY_PATTERN_B_TEMPLATE = r"\{id=\d+,\s*name=([^,}]+?),\s*displayName=[^,}]*?,\s*parentId=\d+,\s*parentName=__PARENT__\}"


def parse_entity_aliases(text):
    """Parse the "EntityName: alias1, alias2" textarea syntax into
    {entity_name: [alias, ...]} (insertion order preserved). Blank lines and
    lines without a colon are skipped. An entity with nothing after the colon
    falls back to its own name as the sole alias."""
    entities = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, aliases_str = line.partition(":")
        name = name.strip()
        if not name:
            continue
        aliases = [a.strip() for a in aliases_str.split(",") if a.strip()]
        entities[name] = aliases or [name]
    return entities


def compile_alias_patterns(entities):
    """{entity_name: [alias, ...]} -> {entity_name: compiled_regex}."""
    compiled = {}
    for name, aliases in entities.items():
        parts = [r"\b" + re.escape(a) for a in aliases]
        compiled[name] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


def detect_candidate_entities(text, compiled_patterns):
    """Which configured entities does this text plausibly mention? Recall-first
    on purpose — the LLM is expected to resolve false positives as 'Not
    Mentioned' or Neutral, not this regex."""
    if not text:
        return []
    return [name for name, rx in compiled_patterns.items() if rx.search(text)]


def entities_from_category_details(category_details, parent_category):
    """Extract entity names BW already tagged under `parent_category` from a
    row's raw 'Category Details' string. Deduped, order preserved."""
    if not category_details or not parent_category:
        return []
    parent_re = re.escape(parent_category)
    pattern_a = re.compile(_CATEGORY_PATTERN_A_TEMPLATE.replace("__PARENT__", parent_re))
    pattern_b = re.compile(_CATEGORY_PATTERN_B_TEMPLATE.replace("__PARENT__", parent_re))
    s = str(category_details)
    found = pattern_a.findall(s) + pattern_b.findall(s)
    uniq = set()
    result = []
    for name in found:
        name = name.strip()
        if name and name not in uniq:
            uniq.add(name)
            result.append(name)
    return result


def discover_category_entities(category_details_values, parent_category):
    """Given every row's raw 'Category Details' value, return {entity_name:
    count} for everything found under parent_category — used to preview the
    entity roster before a run, since (unlike manual aliases) the category
    path doesn't have a fixed list until the data is scanned."""
    counts = Counter()
    for val in category_details_values:
        for name in entities_from_category_details(val, parent_category):
            counts[name] += 1
    return dict(counts)
