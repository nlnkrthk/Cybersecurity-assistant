# Metadata filtering

Hybrid search can optionally restrict eligible incidents before BM25 and
vector retrieval. The filters do not alter incident text, embeddings, BM25
scoring, or the RRF formula.

## Available fields

The current `data/processed/incidents_100k.parquet` contains these populated
metadata fields:

| Field | Non-null rows | Distinct values |
| --- | ---: | ---: |
| `label` | 95,477 | 12 |
| `dst_port` | 95,477 | 11,081 |

Although the Parquet schema contains `src_ip`, `dst_ip`, `src_port`,
`timestamp`, and `protocol`, all their values are null in this dataset. They
are not filterable. Attempting to filter on one returns an error listing the
available fields. No protocol values are inferred.

`attack_type` is accepted as an alias for `label`, and `destination_port` as an
alias for `dst_port`. The canonical schema uses `label` and `dst_port`.

## Usage

Run a single query with one or more `FIELD=VALUE` filters:

```bash
python scripts/search/interactive_hybrid_search.py \
  --filter label=PortScan \
  --filter dst_port=80 \
  --debug \
  "PortScan SYN traffic"
```

Repeated filters are combined with AND semantics. Numeric fields such as
`dst_port` are validated and coerced to their stored numeric type. Unsupported
fields, malformed filters, and invalid values produce clear errors.

## Retrieval behavior

The filter module validates the request against structured index metadata and
calculates eligible incident IDs. BM25 ranks only eligible IDs; Chroma applies
the same conditions using its native metadata `where` filter. RRF combines only
the filtered result lists. A request with no matching records returns empty
result lists.

With `--debug`, the search output reports the normalized filters, total
incidents, eligible incidents, and returned counts for BM25, vector, and RRF.
The result object also includes structured metadata for fused incident IDs.
