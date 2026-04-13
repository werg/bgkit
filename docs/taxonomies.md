# External Taxonomy Builders

This document covers the two scripts that produce **hierarchical** tag paths
for datasets whose own metadata is flat. The Phase 2 browse-tree builder
(`scripts/build_browse_tree.py`) supports two input modes: (1) flat tags
from the Phase 2 mmap's `tag_list_json`, which produces a one-level tree,
or (2) a JSONL file of `(article_id, tag_path)` rows where `tag_path` is
already an ordered root-to-leaf path. For KILT Wikipedia and PubMedQA we
want deep trees, so we use mode (2) and supply the hierarchy from an
external source.

The two builders live in:

- `scripts/build_mesh_hierarchy.py` — PubMedQA articles keyed by
  PubMed ID, hierarchy sourced from the NLM MeSH tree.
- `scripts/build_kilt_hierarchy.py` — KILT Wikipedia articles keyed by
  `wikipedia_id`, hierarchy sourced from the DBpedia SKOS category dump.

Both scripts write JSONL:
```json
{"article_id": "21645374", "tag_path": ["Phenomena and Processes", "Cell Physiological Phenomena", ...]}
```
That file can be fed directly to `build_browse_tree.py --input`.

The scripts are additive — they never touch the Phase 2 mmap, the browse
tree parquet, or any other on-disk artifact. They only *produce* a JSONL
file. Re-running them is safe and idempotent.

## 1. PubMedQA — MeSH tree

### Source

- NLM MeSH ASCII tree file: `mtrees2025.bin` from
  `https://nlmpubs.nlm.nih.gov/projects/mesh/2025/meshtrees/mtrees2025.bin`
- 2.7 MB download, ~65,000 tree numbers, ~31,000 unique MeSH terms.
- NLM announced the ASCII serialization is being discontinued after 2025,
  but the 2025 file is still available under `/projects/mesh/2025/meshtrees/`.
  For future years, update `--mesh-year` or point `--mesh-file` at the
  replacement format (XML/RDF) after extending the parser.

### File format

Each line is `<Term Name>;<Tree Number>` with a single leading space, e.g.
```
 Diseases;C
 Cardiovascular Diseases;C14
 Heart Diseases;C14.280
 Myocardial Ischemia;C14.280.647
 Coronary Disease;C14.280.647.250
```
Tree numbers are dot-separated: each dot adds a level. The first
character encodes the top-level MeSH category (A–Z). A term can have
multiple tree numbers (MeSH is a DAG), and multiple terms can share the
same tree number when the file is updated across revisions — but the
simple `tree_number -> term` map handles this fine.

### Top-level categories

The 16 MeSH letters are mapped to their canonical names via the
`_MESH_LETTER_ROOTS` constant in the script:

```
A: Anatomy                              N: Health Care
B: Organisms                            V: Publication Characteristics
C: Diseases                             Z: Geographicals
D: Chemicals and Drugs                  ...
```

These are the standard NLM names and do not need updating unless NLM
rearranges the MeSH headings (they essentially never do).

### Leaf selection

PubMedQA articles typically carry 15–25 MeSH terms, which include both
the primary subject (e.g. *Myocardial Ischemia*) and demographic /
species tags (*Humans*, *Adult*, *Animals*). A naive "pick the deepest
tree number" wins every time for *Humans* (B01.050.150.900.649.313.988.400.112.400.400 — 11 levels!) and drowns out the real topic. The builder ranks candidates by a
two-level priority:

1. **Letter priority** (`_LETTER_PRIORITY`): Diseases (C) > Chemicals (D) >
   Psychiatry (F) > Techniques (E) > Phenomena (G) > Anatomy (A) > Health
   Care (N) > ... > **Organisms (B) and Named Groups (M) last**.
2. **Tree depth**, largest first, among the highest-priority letter.

On the `pqa_labeled` split (1000 docs), this yields disease/chemical/phenomenon
paths for essentially every article. No articles fell through to the
fallback.

### Running

```bash
# Smoke test (1000 labeled articles, ~10s):
python scripts/build_mesh_hierarchy.py \
    --output /tmp/pubmedqa_labeled.jsonl \
    --splits labeled

# Full run (labeled + artificial, ~211K articles):
python scripts/build_mesh_hierarchy.py \
    --output $DATA_DIR/taxonomies/pubmedqa_hierarchy.jsonl \
    --splits labeled,artificial

# Smaller local copy of the mesh file:
python scripts/build_mesh_hierarchy.py \
    --output out.jsonl --mesh-file /path/to/mtrees2025.bin
```

## 2. KILT Wikipedia — DBpedia SKOS category DAG

### Source

The authoritative Wikipedia category→parent relationship lives in the
`enwiki-latest-categorylinks.sql.gz` dump (~8 GB compressed). That's
large and awkward to parse, so we use a pre-processed surrogate: the
**DBpedia SKOS categories** dump, which serializes the Wikipedia category
system as RDF N-triples using the `skos:broader` (parent-of) and
`skos:prefLabel` (human-readable name) vocabulary. The 2022-12-01 release
for English is 52 MB compressed:

- `https://downloads.dbpedia.org/repo/dbpedia/generic/categories/2022.12.01/categories_lang=en_skos.ttl.bz2`
  (52 MB, 9M lines, 4.5M `broader` edges, 2.2M distinct categories)

DBpedia has newer releases on the Databus but this 2022 version is
available via plain HTTP and covers the vast majority of Wikipedia
categories referenced by KILT (which itself was prepared in 2020–2021).

### KILT Wikipedia dataset

- `orionweller/kilt_wikipedia_split` — a paragraph-split mirror of the
  official `facebook/kilt_wikipedia` dataset. Single JSONL file,
  `train_0.jsonl`, 2.16 GB.
- **~184,094 unique articles** across 1.65M rows. Note that this is a
  *subset* of the full KILT Wikipedia (which has ~5.9M articles); the
  orionweller mirror is the only conveniently-accessible version on HF
  (facebook/kilt_wikipedia uses a deprecated loading script and has no
  published parquet mirror). Our `scripts/convert_hf_to_mmap.py` also
  uses this mirror, so the hierarchy covers 100% of what will land in
  the Phase 2 mmap.
- Each row has `wikipedia_id` (the `document_id` in the Phase 2 mmap),
  `wikipedia_title`, `text`, and `categories` — a comma-separated string
  like `"Academy Awards,Best Art Direction Academy Award winners"`.
- Because the dataset is paragraph-split, the same `wikipedia_id` appears
  in ~9 consecutive rows on average (up to ~650 for long articles). The
  builder de-dupes on the fly and emits each ID once.

### Algorithm

1. Download (or reuse) the SKOS .ttl.bz2 file.
2. Stream-parse the triples into three maps:
   - `child_uri -> set[parent_uri]` via `skos:broader`.
   - `uri -> prefLabel` via `skos:prefLabel`.
   - `lowercase_label -> uri` (inverse of prefLabel; first-wins on
     collisions).
3. Build the inverse adjacency `parent_uri -> [child_uri...]` and BFS
   **from `Category:Main_topic_classifications`** (the canonical top of
   the Wikipedia category DAG, which has 39 direct children: *Academic
   disciplines, Business, Culture, Entertainment, ..., Technology*).
   Stop at depth `--max-depth` (default 6). This gives every reachable
   category a shortest root-path, stored as an ordered list of
   prefLabels.
4. Download (or reuse) `train_0.jsonl`. For each article:
   - Split `categories` on commas.
   - For each raw category name, normalize to lowercase and look up in
     `label_to_uri`. Falls back to a synthetic URI
     (`Category:<underscored_name>`) if the label isn't in the map.
   - If the resolved URI has a cached root path, use it directly.
     Otherwise walk upward through `skos:broader` until an ancestor with
     a cached root path is found, then extend that root path with the
     intermediate labels (this rescues categories like "1995 albums"
     whose own `broader` edge is missing in DBpedia 2022 but whose
     parents reach the root). This is the `_find_extended_path` helper.
   - Pick the **shortest** root path among the article's resolved
     candidates (closest to a top-level bucket, so articles anchor near
     the stable part of the taxonomy; within ties, alphabetical for
     determinism).
5. Write `{"article_id": wikipedia_id, "tag_path": [root->leaf]}`.

If no category matches, write `["Wikipedia", "Uncategorized"]` as a
fallback. The builder tracks two kinds of fallback separately:
- **empty**: the KILT row had no `categories` at all (disambiguation
  pages, stubs, etc.).
- **unmatched**: the categories existed but none were reachable from
  `Main_topic_classifications` within the max depth (rare; usually means
  DBpedia doesn't know about that category yet).

### Why BFS, not full path enumeration

The Wikipedia category DAG has millions of edges and extensive cycles
via "By country", "By year", "Stub" subcategories. A full pathfinding
search would be slow and produce semantically weak paths that go through
arbitrary intermediate grouping categories. BFS from the root gives the
*shortest* path, which tends to route through the semantically important
high-level categories (Science, History, Entertainment, etc.) rather
than through noise categories.

The `--max-depth` of 6 keeps paths interpretable and keeps the BFS
bounded. Typical articles end up with paths of length 4–7 levels
including the root.

### Running

```bash
# Smoke test (200 articles, ~90s, mostly SKOS parse):
python scripts/build_kilt_hierarchy.py \
    --output /tmp/kilt_sample.jsonl --limit 200

# Full run (5.9M articles, ~10-20 min):
python scripts/build_kilt_hierarchy.py \
    --output $DATA_DIR/taxonomies/kilt_wikipedia_hierarchy.jsonl

# Reuse already-downloaded files:
python scripts/build_kilt_hierarchy.py \
    --output out.jsonl \
    --skos-file /tmp/kilt_tax/skos.ttl.bz2 \
    --kilt-jsonl /tmp/kilt_tax/train_0.jsonl

# Prefer streaming over downloading the full 2.16 GB JSONL:
python scripts/build_kilt_hierarchy.py \
    --output out.jsonl --skip-kilt-download
```

## Downstream use

Both outputs are consumed by:

```bash
python scripts/build_browse_tree.py \
    --dataset kilt_wikipedia \
    --input $DATA_DIR/taxonomies/kilt_wikipedia_hierarchy.jsonl \
    --output-dir $DATA_DIR/browse_trees

python scripts/build_browse_tree.py \
    --dataset pubmedqa \
    --input $DATA_DIR/taxonomies/pubmedqa_hierarchy.jsonl \
    --output-dir $DATA_DIR/browse_trees
```

The `BrowseTreeBuilder` in `src/bgkit/data/tagging.py` handles the rest:
it materializes every tag-path prefix as an internal node, subdivides
oversized leaves into alphabetical buckets, and caps fanout to keep the
navigation UX sane.

Note that `article_id` in the JSONL **must** match `document_id` in the
corresponding Phase 2 mmap's `metadata.parquet` (so the browse tree can
resolve back to the actual tokens). For PubMedQA this is the PubMed ID
(`pubid` in the raw HF dataset), and for KILT Wikipedia this is the
`wikipedia_id`.

**Related convert_hf_to_mmap.py change (2026-04-11):** the Phase 2
converter's `document_id` fallback chain previously only checked
`document_id → wikipedia_id → id → idx`, which meant PubMedQA and a few
other datasets fell through to an opaque enumeration index. The chain
now probes `document_id → wikipedia_id → pubid → query_id → key → id
→ idx`, matching the HF field names that each dataset actually uses.
This is a *fix*, not a break: no Phase 2 mmap had been built yet when
the change landed, so there's no existing on-disk state to migrate.
The hierarchy JSONL files produced by these builders will match the
`document_id` column of any Phase 2 mmap built after this change.

## Tradeoffs and open questions

### MeSH

- **Demographic MeSH terms are down-ranked.** The letter priority makes
  every article anchor to its disease/chemical/phenomenon subject rather
  than *Humans* or *Animals*. Articles that are genuinely about a
  species (veterinary studies, pure organism biology) will still appear
  under *Diseases* or *Chemicals* if any such MeSH term is present. If
  that becomes a problem, `_LETTER_PRIORITY` can be tuned or made
  per-split.
- **No multi-path attachment.** The current design picks one canonical
  path per article. A MeSH article often has multiple valid trees
  (e.g., under both Diseases and Chemicals). The browse tree builder
  supports multi-tag articles but we keep to one path for simplicity and
  because the Phase 2 KR retriever downstream only needs one stable
  bucket per document.
- **Revision drift.** The 2025 MeSH file differs from PubMedQA's original
  2019-era MeSH tagging. Terms that have been merged or renamed won't be
  in the current tree and will be silently skipped. The 4–8 % of
  PubMedQA meshes that don't match the 2025 tree all still have at least
  one other mesh that does match, so the pick-one heuristic always
  produces a path.

### KILT

- **DBpedia is a 2022 snapshot.** KILT itself was built from a 2019
  Wikipedia dump, and DBpedia 2022 is a 2022 snapshot. Most categories
  overlap but not all — some categories deleted between 2019 and 2022
  won't be in the DAG and will hit the unmatched-fallback path. For the
  200-article smoke test, the observed empty-vs-unmatched split was
  35 fallbacks, all due to the KILT row having an empty
  `categories` field (disambiguation pages), not due to DBpedia drift.
- **Root is `Main_topic_classifications`, not the article letter.**
  Wikipedia doesn't have a single "root" in the DAG; we pick
  `Main_topic_classifications` because it's the curated top-level set of
  39 subject buckets that Wikipedia itself uses. Articles with
  categories entirely outside this subtree (stub/maintenance/by-country
  categories) will hit unmatched-fallback.
- **Shortest-path selection.** Within an article's reachable
  categories, the builder picks the *shortest* root path, not the
  longest. This was a deliberate call: Wikipedia's DAG has many deep
  chains through arbitrary "By country" / "By year" groupings that are
  not semantically useful, and shortest paths tend to route through the
  canonical high-level categories (Science, Entertainment, History, ...)
  that make a good browsing hierarchy.
- **Memory footprint.** The SKOS parse holds ~2.2 M URI strings and
  ~4.5 M edges in memory. Measured peak ~1.5 GB RSS on the DGX Spark.
  Should be fine anywhere but a constrained laptop.

## Coverage summary

Results from the end-to-end runs on 2026-04-11:

| Dataset | Articles | Hierarchical | Fallback (empty) | Fallback (unmatched) |
|---|---|---|---|---|
| PubMedQA (labeled) | 1,000 | 1,000 (100.0%) | 0 | 0 |
| PubMedQA (labeled+artificial) | 212,269 | 201,197 (94.8%) | 11,072 | 0 |
| KILT Wikipedia | 184,094 | 154,218 (83.8%) | 22,912 | 6,964 |

**PubMedQA fallbacks** are all articles whose `context.meshes` list is
empty in the raw HF data — not a builder issue. Every article with at
least one MeSH term got a deep hierarchical path.

**KILT fallbacks** split two ways:
- **Empty (22,912 = 12.4%)**: article has no `categories` field in the
  source data. Disambiguation pages, stubs, and maintenance redirects.
  Unavoidable.
- **Unmatched (6,964 = 3.8%)**: article has categories but none could
  be routed to `Main_topic_classifications` via the `skos:broader`
  graph, even after the ancestor walk in `_find_extended_path`. These
  are typically articles in very recent or meta categories that DBpedia
  2022 hasn't indexed, or categories DBpedia marks as isolated leaves
  (orphans) whose entire parent chain is also orphaned.

Top-level KILT buckets by article count (top 10):

```
  29,928  Geography
  18,501  People
  13,346  Entertainment
  10,911  Sports
   8,872  Government
   8,756  Culture
   7,019  Mass media
   5,448  Life
   5,337  Academic disciplines
   5,107  Business
```

The distribution reaches all 39 Main-topic-classification buckets (down
to "Universe" with 12 articles). Depths range from 2 (fallback) to 13
levels, with median 6 — giving the browse tree meaningful navigation
depth without excessive chain length.

Top-level PubMedQA buckets:

```
  154,464  Diseases
   35,346  Chemicals and Drugs
    5,419  Psychiatry and Psychology
    5,262  Analytical, Diagnostic and Therapeutic Techniques, and Equipment
      531  Phenomena and Processes
      128  Health Care
       32  Anatomy
  ...
```

## Generated artifacts

Both JSONL files are on the external drive at:

- `$DATA_DIR/taxonomies/pubmedqa_hierarchy.jsonl` (38 MB, 212K rows)
- `$DATA_DIR/taxonomies/kilt_wikipedia_hierarchy.jsonl` (33 MB, 184K rows)

Downloaded intermediates (safe to delete, will be re-fetched on next
run):

- `/tmp/mesh/mtrees2025.bin` (2.7 MB)
- `/tmp/kilt_tax/skos.ttl.bz2` (52 MB)
- `/tmp/kilt_tax/train_0.jsonl` (2.16 GB)
