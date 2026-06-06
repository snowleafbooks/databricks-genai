# Install

End-to-end setup for running the book's notebooks in your own Databricks workspace.

**Nothing runs on your local machine** — no CLI install, no `git clone`, no manual file upload. Everything happens inside Databricks.

## 1. Databricks workspace prerequisites

You need a workspace where you can:

- Create a Unity Catalog (UC) catalog and Volume, or use an existing one
- Use **default serverless** compute (no need to provision a cluster)
- Reach the **Foundation Model APIs** (pay-per-token tier is enough for ~90% of the book)
- Create **Vector Search** endpoints (Ch 8 onward)
- Create **Model Serving** endpoints with the **AI Gateway** (Ch 10 onward)
- Reach `github.com` over HTTPS (the Chapter 0 notebook downloads the bootstrap data from this public repo)

Workspaces on AWS / Azure / GCP all work — the book uses `docs.databricks.com/aws/en/...` URLs as the canonical doc reference, but the underlying APIs are cloud-agnostic.

If your workspace has no outbound HTTPS to github.com (locked-down enterprise networks), see **Offline workspaces** at the bottom of this file.

## 2. Pick your catalog name

The book defaults to `genaicert` as the UC catalog. If you'd rather use a different name (`my_book_genai`, etc.), pick it now — every notebook accepts a `catalog` widget so you only have to override at the widget, never in the notebook source.

This INSTALL assumes `genaicert`; substitute your choice as needed.

## 3. Bring the Chapter 0 notebook into your workspace

Pick whichever is easier on your workspace:

**Option A — Import the notebook URL.** In the Databricks UI, go to **Workspace → Create → Import**, choose **URL**, and paste:

```
https://raw.githubusercontent.com/snowleafbooks/databricks-genai/main/notebooks/00-setup/c0001-upload-to-databricks.ipynb
```

The notebook lands in your personal folder.

**Option B — Mount the entire companion repo as a Databricks Git folder.** In the workspace, **Workspace → Git folders → Create**, paste `https://github.com/snowleafbooks/databricks-genai`. The whole `notebooks/` tree (Ch 0, 3–15) appears as workspace files at `/Repos/.../databricks-genai/notebooks/`. This is the more convenient option if you plan to work through the whole book — you get all chapters at once rather than importing one at a time.

## 4. Run the setup notebook

Open `c0001-upload-to-databricks` in your workspace. Two widgets at the top:

| Widget | Default | Override when |
|---|---|---|
| `catalog` | `genaicert` | Your workspace uses a different catalog name |
| `data_source` | GitHub archive URL | Your workspace has no outbound HTTPS — see *Offline workspaces* below |

Click **Run all** on default serverless compute. The notebook:

1. Downloads `volume-bootstrap/` from the GitHub archive (~7 MB)
2. Creates UC catalog (`genaicert`), three schemas (`pawshield`, `eval`, `monitoring`), and the Volume (`pawshield.bootstrap`)
3. Stages the data into the Volume (7 parquet tables + six file-artifact subdirs)
4. Loads 7 Delta tables from the staged parquet
5. Verifies Sarah Chen's lifecycle anchor (her email, her policy PDF, her vet invoice, her claim row)

Total runtime: ~2–3 minutes.

If the verification assertions all pass, you're ready for Chapter 1.

## 5. (Optional) Provision Vector Search + Model Serving

These are needed for Ch 8 onward:

- **Vector Search endpoint** named `pawshield-vs` — Ch 8's builder notebook (`c0801-build-policy-index.ipynb`) creates the endpoint + a Delta-Sync index over `policy_chunks` automatically. Allow ~10 min for the endpoint to come online the first time.
- **Model Serving endpoints** — Ch 10 (`c1001-deploy-policypal.ipynb`) creates a Custom Model Serving endpoint for PolicyPal. Ch 11/13/14 reuse it.

If your workspace's Vector Search endpoints get torn down between sessions, just re-run `c0801-build-policy-index.ipynb` to re-provision before Ch 9 / Ch 13.

## Troubleshooting

**`PERMISSION_DENIED` creating the catalog** — your workspace admin needs to grant you `CREATE CATALOG` on the metastore, or create `genaicert` for you and give you `OWNERSHIP`.

**Cells fail with rate-limit (HTTP 429)** — the Foundation Model APIs enforce per-workspace token-per-minute caps. Notebooks include a 0.3s throttle + retry pattern; if the cell still fails repeatedly, the workspace's pay-per-token quota is saturated. Ch 6 + Ch 10 walk the provisioned-throughput escape hatch.

**Vector Search endpoint stuck in `PROVISIONING`** — first-time endpoint creation can take 15+ minutes. Watch the status in **Catalog Explorer → Vector Search**. If it stays stuck > 30 min, contact your workspace admin.

**Ch 0 fails downloading the GitHub archive** — your workspace can't reach github.com. See *Offline workspaces* below.

## Offline workspaces

If your workspace has no outbound HTTPS to github.com (locked-down enterprise networks), an admin needs to pre-stage the bootstrap data once, then every reader sets the `data_source` widget to that pre-staged Volume path.

The admin:

1. Downloads `volume-bootstrap/` from this repo (e.g., by downloading the repo's `Code → Download ZIP` button on a machine that has internet access).
2. Uploads `volume-bootstrap/` to a Volume on the locked-down workspace — typically `/Volumes/genaicert/pawshield/bootstrap` — via Databricks Volume UI upload, or via whatever in-workspace file-transfer mechanism the org provides.

Each reader then sets the `data_source` widget on Chapter 0 to that Volume path before clicking Run all. The notebook detects the same-path case and skips the staging copy (it just creates the catalog/schemas/Volume entry and loads the parquet tables that are already there).

## What if I don't have a Databricks workspace?

Every notebook in this repo ships with cell outputs preserved — open any `.ipynb` on GitHub and you'll see the figures, dataframes, traces, and error messages from a real run. You won't be able to *modify and re-run*, but you can follow the chapter prose against the rendered outputs and learn the patterns without provisioning anything.

The book itself is the primary teaching surface; the notebooks are the verifiable reference implementation.
