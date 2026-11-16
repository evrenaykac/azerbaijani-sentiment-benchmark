# Access to the comment text

The public release contains comment identifiers, labels, the fixed split,
per-annotator records, source-group hashes and a hash of the deduplication key
for all 271,278 comments. It does not contain the comment text: the comments
were collected from the public comment sections of YouTube, Instagram and
Facebook pages and from the review sections of application stores and
e-commerce sites, and the terms of these platforms do not allow us to republish
user content.

## What needs the text and what does not

| Purpose | Needs the text |
|---|---|
| Recomputing every experimental result of the article from the stored predictions and the index | no |
| Recomputing the agreement statistics, per batch and per cohort, from `data/annotations.csv` | no |
| Verifying the duplicate-text invariants through `text_key_sha256` | no |
| Training or evaluating a new system on the fixed split | yes |

## Route 1: on request

The text is shared for non-commercial research under a data-use agreement.
To request it, write to the authors (see *Contact* in the README; the address of
the corresponding author is given in the article) with your name, affiliation
and a short description of the intended use. The agreement carries
the restrictions under which the authors work:

1. the text is used for non-commercial research only;
2. the text is not redistributed, in whole or in part, and neither are derived
   files from which it can be reconstructed;
3. no attempt is made to identify or contact the writers of the comments.

After the agreement is signed you receive a file that maps each `uid` of
`data/corpus_index.csv.gz` to its text.

## Route 2: regeneration from your own copy

If you hold a copy of the source material, the text can be rejoined without
contacting the authors:

```bash
python scripts/regenerate_text.py --source your_comments.csv --text-column text
```

The script recomputes the deduplication key of each comment and joins it to the
index through `text_key_sha256`.
