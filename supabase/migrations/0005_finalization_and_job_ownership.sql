-- Migration 0005: finalization and job ownership fixes

UPDATE public.chat_jobs SET error = '' WHERE error IS NULL;

ALTER TABLE public.chat_jobs ALTER COLUMN error SET DEFAULT '';
ALTER TABLE public.chat_jobs ALTER COLUMN error SET NOT NULL;

CREATE INDEX IF NOT EXISTS chat_jobs_conversation_active_idx ON public.chat_jobs (conversation_id) WHERE status IN ('queued','running');

CREATE OR REPLACE FUNCTION public.create_chat_job(
  _conversation_id uuid,
  _job_id uuid,
  _client_id text,
  _title text,
  _messages jsonb,
  _history jsonb
)
RETURNS TABLE(conversation_id uuid, job_id uuid, created boolean)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  existing_owner text;
  conversation_exists boolean := false;
  now_ts timestamptz := pg_catalog.clock_timestamp();
BEGIN
  SELECT conversations.client_id
  INTO existing_owner
  FROM public.chat_conversations AS conversations
  WHERE conversations.id = _conversation_id
  FOR UPDATE;

  IF FOUND THEN
    conversation_exists := true;
  ELSE
    BEGIN
      INSERT INTO public.chat_conversations (
        id,
        title,
        messages,
        client_id,
        created_at,
        updated_at
      )
      VALUES (
        _conversation_id,
        _title,
        coalesce(_messages, '[]'::jsonb),
        _client_id,
        now_ts,
        now_ts
      );
      conversation_exists := false;
    EXCEPTION WHEN unique_violation THEN
      SELECT conversations.client_id
      INTO existing_owner
      FROM public.chat_conversations AS conversations
      WHERE conversations.id = _conversation_id
      FOR UPDATE;

      IF NOT FOUND THEN
        RAISE EXCEPTION 'A beszélgetés nem hozható létre.'
          USING ERRCODE = '40001';
      END IF;

      conversation_exists := true;
    END;
  END IF;

  IF conversation_exists THEN
    IF existing_owner IS NOT NULL AND existing_owner <> _client_id THEN
      RAISE EXCEPTION 'A beszélgetés más ügyfélhez tartozik.'
        USING ERRCODE = '42501';
    END IF;

    UPDATE public.chat_conversations AS conversations
    SET title = CASE
          WHEN conversations.title IS NULL OR conversations.title = '' THEN _title
          ELSE conversations.title
        END,
        messages = coalesce(_messages, conversations.messages),
        client_id = coalesce(conversations.client_id, _client_id),
        updated_at = now_ts
    WHERE conversations.id = _conversation_id;
  END IF;

  -- Cancel previous running jobs and queued jobs for this conversation
  UPDATE public.chat_jobs
  SET stop_requested = true,
      updated_at = now_ts
  WHERE conversation_id = _conversation_id
    AND status = 'running';

  UPDATE public.chat_jobs
  SET status = 'cancelled',
      finished_at = now_ts,
      lease_expires_at = NULL,
      updated_at = now_ts
  WHERE conversation_id = _conversation_id
    AND status = 'queued';

  INSERT INTO public.chat_jobs (
    id,
    conversation_id,
    history,
    status,
    created_at,
    updated_at
  )
  VALUES (
    _job_id,
    _conversation_id,
    coalesce(_history, '[]'::jsonb),
    'queued',
    now_ts,
    now_ts
  );

  RETURN QUERY
    SELECT _conversation_id AS conversation_id,
           _job_id AS job_id,
           NOT conversation_exists AS created;
END;
$$;

REVOKE ALL ON FUNCTION public.create_chat_job(uuid, uuid, text, text, jsonb, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.create_chat_job(uuid, uuid, text, text, jsonb, jsonb) TO service_role;

REVOKE ALL ON FUNCTION public.kick_chat_worker(text, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.kick_chat_worker(text, uuid) TO service_role;

NOTIFY pgrst, 'reload schema';
