"""T-008 acceptance: the eval queries must return the right manual in the top hits."""

import sys

from lab_memory_mcp import retrieval

QUERIES = [
    ("How do I assign an expression pedal on the MOOD MKII?", "MOOD MKII"),
    ("What MIDI CC controls the ramp on Generation Loss MKII?", "Generation Loss"),
    ("How do the dip switches change clock behavior on MOOD?", "MOOD"),
    ("How do I do a factory reset on Generation Loss MKII?", "Generation Loss"),
]

def main() -> None:
    client = retrieval.milvus()
    stats = client.get_collection_stats(retrieval.COLLECTION)
    print(f"COLLECTION {retrieval.COLLECTION} chunks={stats.get('row_count')}\n")
    passed = 0
    for query, expect in QUERIES:
        hits = retrieval.recall(query, k=3, tag="gear", expand=False)
        titles = [h.title for h in hits]
        ok = any(expect.lower() in t.lower() for t in titles[:3])
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} :: {query}")
        for h in hits:
            print(f"   [{h.score}] {h.title}  (chunk {h.chunk_ix})")
            print(f"          {h.snippet[:160].replace(chr(10),' ')}...")
            print(f"          {h.citation}")
        print()
    print(f"RESULT {passed}/{len(QUERIES)} eval queries returned the expected manual in top-3")
    sys.exit(0 if passed == len(QUERIES) else 1)

if __name__ == "__main__":
    main()
