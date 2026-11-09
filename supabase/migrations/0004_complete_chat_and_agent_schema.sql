CREATE TABLE IF NOT EXISTS public.chat_conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id text,
  title text NOT NULL DEFAULT '',
  messages jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL,
  history jsonb NOT NULL DEFAULT '[]'::jsonb,
  status text NOT NULL DEFAULT 'queued',
  response_content text NOT NULL DEFAULT '',
  response_reasoning text NOT NULL DEFAULT '',
  error text NOT NULL DEFAULT '',
  attempts integer NOT NULL DEFAULT 0,
  next_attempt_at timestamptz,
  last_error text,
  progress_at timestamptz,
  checkpoint jsonb,
  started_at timestamptz,
  finished_at timestamptz,
  lease_expires_at timestamptz,
  stop_requested boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_job_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  job_id uuid NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_skills (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  dsl text NOT NULL,
  terms text NOT NULL DEFAULT '',
  vector jsonb NOT NULL DEFAULT '[]'::jsonb,
  successes integer NOT NULL DEFAULT 0,
  failures integer NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'active',
  baseline_rate double precision NOT NULL DEFAULT 0,
  candidate_rate double precision NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_playbooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  content text NOT NULL,
  task_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_policy_weights (
  ngram text PRIMARY KEY,
  weight double precision NOT NULL DEFAULT 0,
  observations integer NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_tasks (
  id text PRIMARY KEY,
  objective text NOT NULL DEFAULT '',
  outcome text NOT NULL DEFAULT '',
  steps integer NOT NULL DEFAULT 0,
  verified boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_reflections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id text NOT NULL DEFAULT '',
  objective text NOT NULL DEFAULT '',
  failure_point text NOT NULL DEFAULT '',
  component text NOT NULL DEFAULT '',
  root_cause text NOT NULL DEFAULT '',
  pivot_action text NOT NULL DEFAULT '',
  patch text NOT NULL DEFAULT '',
  failed_action text NOT NULL DEFAULT '',
  corrective_action text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_operator_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id text NOT NULL,
  content jsonb NOT NULL,
  consumed boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_browser_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL,
  browser_id text NOT NULL UNIQUE,
  cdp_url text NOT NULL,
  live_url text,
  control_mode text NOT NULL DEFAULT 'agent',
  status text NOT NULL DEFAULT 'active',
  agent_busy boolean NOT NULL DEFAULT false,
  viewport_width integer NOT NULL DEFAULT 1280,
  viewport_height integer NOT NULL DEFAULT 1029,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.chat_conversations
  ADD COLUMN IF NOT EXISTS client_id text,
  ADD COLUMN IF NOT EXISTS title text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS messages jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.chat_jobs
  ADD COLUMN IF NOT EXISTS conversation_id uuid,
  ADD COLUMN IF NOT EXISTS history jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'queued',
  ADD COLUMN IF NOT EXISTS response_content text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS response_reasoning text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS error text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz,
  ADD COLUMN IF NOT EXISTS last_error text,
  ADD COLUMN IF NOT EXISTS progress_at timestamptz,
  ADD COLUMN IF NOT EXISTS checkpoint jsonb,
  ADD COLUMN IF NOT EXISTS started_at timestamptz,
  ADD COLUMN IF NOT EXISTS finished_at timestamptz,
  ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz,
  ADD COLUMN IF NOT EXISTS stop_requested boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.chat_job_events
  ADD COLUMN IF NOT EXISTS job_id uuid,
  ADD COLUMN IF NOT EXISTS payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.chat_browser_sessions
  ADD COLUMN IF NOT EXISTS agent_busy boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS viewport_width integer NOT NULL DEFAULT 1280,
  ADD COLUMN IF NOT EXISTS viewport_height integer NOT NULL DEFAULT 1029,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_skills
  ADD COLUMN IF NOT EXISTS name text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS dsl text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS terms text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS vector jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS successes integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS failures integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS baseline_rate double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS candidate_rate double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_playbooks
  ADD COLUMN IF NOT EXISTS content text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS task_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_policy_weights
  ADD COLUMN IF NOT EXISTS weight double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS observations integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_tasks
  ADD COLUMN IF NOT EXISTS objective text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS outcome text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS steps integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS verified boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_reflections
  ADD COLUMN IF NOT EXISTS task_id text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS objective text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS failure_point text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS component text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS root_cause text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS pivot_action text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS patch text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS failed_action text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS corrective_action text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE public.agent_operator_messages
  ADD COLUMN IF NOT EXISTS task_id text,
  ADD COLUMN IF NOT EXISTS content jsonb NOT NULL DEFAULT '""'::jsonb,
  ADD COLUMN IF NOT EXISTS consumed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

DO $$
DECLARE
  content_type text;
BEGIN
  SELECT columns.udt_name
  INTO content_type
  FROM information_schema.columns AS columns
  WHERE columns.table_schema = 'public'
    AND columns.table_name = 'agent_operator_messages'
    AND columns.column_name = 'content';

  IF content_type IS NOT NULL AND content_type <> 'jsonb' THEN
    ALTER TABLE public.agent_operator_messages
      ALTER COLUMN content DROP DEFAULT;
    ALTER TABLE public.agent_operator_messages
      ALTER COLUMN content TYPE jsonb
      USING COALESCE(pg_catalog.to_jsonb(content), '""'::jsonb);
  END IF;

  UPDATE public.agent_operator_messages
  SET content = '""'::jsonb
  WHERE content IS NULL;

  ALTER TABLE public.agent_operator_messages
    ALTER COLUMN content SET DEFAULT '""'::jsonb;
  ALTER TABLE public.agent_operator_messages
    ALTER COLUMN content SET NOT NULL;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_constraint AS constraints
    WHERE constraints.conrelid = 'public.chat_jobs'::regclass
      AND constraints.conname = 'chat_jobs_conversation_id_fkey'
  ) THEN
    ALTER TABLE public.chat_jobs
      ADD CONSTRAINT chat_jobs_conversation_id_fkey
      FOREIGN KEY (conversation_id)
      REFERENCES public.chat_conversations(id)
      ON DELETE CASCADE
      NOT VALID;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_constraint AS constraints
    WHERE constraints.conrelid = 'public.chat_job_events'::regclass
      AND constraints.conname = 'chat_job_events_job_id_fkey'
  ) THEN
    ALTER TABLE public.chat_job_events
      ADD CONSTRAINT chat_job_events_job_id_fkey
      FOREIGN KEY (job_id)
      REFERENCES public.chat_jobs(id)
      ON DELETE CASCADE
      NOT VALID;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_constraint AS constraints
    WHERE constraints.conrelid = 'public.chat_browser_sessions'::regclass
      AND constraints.conname = 'chat_browser_sessions_job_id_fkey'
  ) THEN
    ALTER TABLE public.chat_browser_sessions
      ADD CONSTRAINT chat_browser_sessions_job_id_fkey
      FOREIGN KEY (job_id)
      REFERENCES public.chat_jobs(id)
      ON DELETE CASCADE
      NOT VALID;
  END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS chat_jobs_conversation_created_idx
  ON public.chat_jobs (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_job_events_job_id_idx
  ON public.chat_job_events (job_id, id);
CREATE INDEX IF NOT EXISTS chat_browser_sessions_job_idx
  ON public.chat_browser_sessions (job_id);
CREATE INDEX IF NOT EXISTS chat_browser_sessions_status_idx
  ON public.chat_browser_sessions (status, job_id);
CREATE INDEX IF NOT EXISTS agent_skills_status_idx
  ON public.agent_skills (status, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_reflections_task_idx
  ON public.agent_reflections (task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_operator_messages_pending_idx
  ON public.agent_operator_messages (task_id, created_at)
  WHERE consumed = false;

ALTER TABLE public.chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_job_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_browser_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_policy_weights ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_reflections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_operator_messages ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.chat_conversations FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.chat_jobs FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.chat_job_events FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.chat_browser_sessions FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_skills FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_playbooks FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_policy_weights FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_tasks FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_reflections FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_operator_messages FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.chat_conversations TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.chat_jobs TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.chat_job_events TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.chat_browser_sessions TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_skills TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_playbooks TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_policy_weights TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_tasks TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_reflections TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_operator_messages TO service_role;

GRANT USAGE, SELECT ON SEQUENCE public.chat_job_events_id_seq TO service_role;

NOTIFY pgrst, 'reload schema';
