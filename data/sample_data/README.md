# Sample data

Small files for trying the system and reviewing it without the full dataset. Regenerate them with
`python scripts/data/make_sample_data.py` (needs the prepared incident library).

| File | Contents | Use it with |
|---|---|---|
| `sample_incidents.csv` | 120 incidents: the first 8 of each of the 15 traffic types in the prepared library. Columns: `incident_id`, `label`, `dst_port`, the 24 model features and the `incident_text` that is indexed for search | Flow Analysis (enter the metrics or the incident ID), `POST /ml/*`, `GET /incidents/{id}`, checking search results |
| `triage_queue_sample.csv` | 14 rows for ranking: 4 historical incident IDs (Heartbleed, SSH-Patator, PortScan, BENIGN), 9 flows given as metrics (DDoS, DoS Hulk, SQL injection, Infiltration, Bot, FTP-Patator, XSS, BENIGN) and one partial flow (duration and packets only) | Triage Queue page (upload), `POST /ml/prioritize/csv` |
| `sample_queries.json` | 8 assistant questions with the facts a correct answer must contain (route, attack type, priority, cited IDs, filters, block or redaction) and an excerpt of the assistant's real answer from the evaluation run | AI Assistant page, `POST /chat` |

The full labelled question set used for evaluation is `src/evaluation/cases/deepeval_cases.json`
(24 cases). The dataset itself is described in `docs/Dataset_Details.pdf`.
