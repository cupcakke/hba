---
name: Supabase migration target verification
description: Keep Supabase migration tooling aligned with the project's live API and PostgreSQL connection.
---

Before applying SQL migrations, verify that the project reference in `supabase/config.toml` matches the runtime `SUPABASE_URL` and that `SUPABASE_DB_URL` points to the same Supabase project. A repository project reference can be stale even while the app's REST API and database credentials work.

**Why:** The configured migration project differed from the Supabase project used by the running app, so the first target check correctly blocked schema changes until the references were reconciled.

**How to apply:** Before any external Supabase DDL, compare the runtime API project reference with the database connection host or username. Do not apply migrations when the targets differ, and do not print connection credentials.
