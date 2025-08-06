--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

-- Started on 2025-08-06 14:24:39 UTC

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 3 (class 3079 OID 24890)
-- Name: btree_gin; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA public;


--
-- TOC entry 3901 (class 0 OID 0)
-- Dependencies: 3
-- Name: EXTENSION btree_gin; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION btree_gin IS 'support for indexing common datatypes in GIN';


--
-- TOC entry 4 (class 3079 OID 25326)
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- TOC entry 3902 (class 0 OID 0)
-- Dependencies: 4
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- TOC entry 5 (class 3079 OID 25407)
-- Name: unaccent; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;


--
-- TOC entry 3903 (class 0 OID 0)
-- Dependencies: 5
-- Name: EXTENSION unaccent; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION unaccent IS 'text search dictionary that removes accents';


--
-- TOC entry 2 (class 3079 OID 24879)
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- TOC entry 3904 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- TOC entry 400 (class 1255 OID 25440)
-- Name: create_system_alert(text, text, text, text, text, jsonb); Type: FUNCTION; Schema: public; Owner: sutazai
--

CREATE FUNCTION public.create_system_alert(alert_type_param text, severity_param text, title_param text, description_param text, source_param text DEFAULT NULL::text, metadata_param jsonb DEFAULT '{}'::jsonb) RETURNS integer
    LANGUAGE plpgsql
    AS $$
    DECLARE
        alert_id INTEGER;
    BEGIN
        INSERT INTO system_alerts (
            alert_type, severity, title, description, source, metadata
        ) VALUES (
            alert_type_param, severity_param, title_param, 
            description_param, source_param, metadata_param
        ) RETURNING id INTO alert_id;
        
        RETURN alert_id;
    END;
    $$;


ALTER FUNCTION public.create_system_alert(alert_type_param text, severity_param text, title_param text, description_param text, source_param text, metadata_param jsonb) OWNER TO sutazai;

--
-- TOC entry 399 (class 1255 OID 25439)
-- Name: log_api_usage(text, text, integer, numeric, integer, integer); Type: FUNCTION; Schema: public; Owner: sutazai
--

CREATE FUNCTION public.log_api_usage(endpoint_param text, method_param text, response_code_param integer, response_time_param numeric, user_id_param integer DEFAULT NULL::integer, agent_id_param integer DEFAULT NULL::integer) RETURNS void
    LANGUAGE plpgsql
    AS $$
    BEGIN
        INSERT INTO api_usage_logs (
            endpoint, method, response_code, response_time, user_id, agent_id
        ) VALUES (
            endpoint_param, method_param, response_code_param, 
            response_time_param, user_id_param, agent_id_param
        );
    END;
    $$;


ALTER FUNCTION public.log_api_usage(endpoint_param text, method_param text, response_code_param integer, response_time_param numeric, user_id_param integer, agent_id_param integer) OWNER TO sutazai;

--
-- TOC entry 398 (class 1255 OID 25438)
-- Name: update_agent_health_status(text, text, numeric, numeric, numeric); Type: FUNCTION; Schema: public; Owner: sutazai
--

CREATE FUNCTION public.update_agent_health_status(agent_name_param text, status_param text, cpu_usage_param numeric DEFAULT NULL::numeric, memory_usage_param numeric DEFAULT NULL::numeric, response_time_param numeric DEFAULT NULL::numeric) RETURNS boolean
    LANGUAGE plpgsql
    AS $$
    DECLARE
        agent_id_var INTEGER;
    BEGIN
        -- Get agent ID
        SELECT id INTO agent_id_var FROM agents WHERE name = agent_name_param;
        
        IF agent_id_var IS NULL THEN
            RETURN FALSE;
        END IF;
        
        -- Insert or update health record
        INSERT INTO agent_health (
            agent_id, status, last_heartbeat, cpu_usage, memory_usage, response_time
        ) VALUES (
            agent_id_var, status_param, CURRENT_TIMESTAMP, 
            cpu_usage_param, memory_usage_param, response_time_param
        )
        ON CONFLICT (agent_id) DO UPDATE SET
            status = EXCLUDED.status,
            last_heartbeat = EXCLUDED.last_heartbeat,
            cpu_usage = COALESCE(EXCLUDED.cpu_usage, agent_health.cpu_usage),
            memory_usage = COALESCE(EXCLUDED.memory_usage, agent_health.memory_usage),
            response_time = COALESCE(EXCLUDED.response_time, agent_health.response_time);
        
        RETURN TRUE;
    END;
    $$;


ALTER FUNCTION public.update_agent_health_status(agent_name_param text, status_param text, cpu_usage_param numeric, memory_usage_param numeric, response_time_param numeric) OWNER TO sutazai;

--
-- TOC entry 265 (class 1255 OID 24852)
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: sutazai
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO sutazai;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 228 (class 1259 OID 24643)
-- Name: agent_executions; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.agent_executions (
    id integer NOT NULL,
    agent_id integer,
    task_id integer,
    status character varying(50),
    input_data jsonb,
    output_data jsonb,
    execution_time double precision,
    error_message text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.agent_executions OWNER TO sutazai;

--
-- TOC entry 227 (class 1259 OID 24642)
-- Name: agent_executions_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.agent_executions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.agent_executions_id_seq OWNER TO sutazai;

--
-- TOC entry 3905 (class 0 OID 0)
-- Dependencies: 227
-- Name: agent_executions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agent_executions_id_seq OWNED BY public.agent_executions.id;


--
-- TOC entry 234 (class 1259 OID 24709)
-- Name: agent_health; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.agent_health (
    id integer NOT NULL,
    agent_id integer,
    status character varying(50) DEFAULT 'unknown'::character varying NOT NULL,
    last_heartbeat timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    cpu_usage numeric(5,2) DEFAULT 0.00,
    memory_usage numeric(5,2) DEFAULT 0.00,
    disk_usage numeric(5,2) DEFAULT 0.00,
    response_time numeric(8,3) DEFAULT 0.000,
    error_count integer DEFAULT 0,
    success_count integer DEFAULT 0,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.agent_health OWNER TO sutazai;

--
-- TOC entry 233 (class 1259 OID 24708)
-- Name: agent_health_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.agent_health_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.agent_health_id_seq OWNER TO sutazai;

--
-- TOC entry 3906 (class 0 OID 0)
-- Dependencies: 233
-- Name: agent_health_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agent_health_id_seq OWNED BY public.agent_health.id;


--
-- TOC entry 222 (class 1259 OID 24591)
-- Name: agents; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.agents (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    type character varying(50) NOT NULL,
    description text,
    endpoint character varying(255) NOT NULL,
    port integer,
    is_active boolean DEFAULT true,
    capabilities jsonb DEFAULT '[]'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.agents OWNER TO sutazai;

--
-- TOC entry 247 (class 1259 OID 24858)
-- Name: agent_status_overview; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.agent_status_overview AS
 SELECT a.id,
    a.name,
    a.type,
    a.is_active AS configured_active,
    ah.status AS health_status,
    ah.last_heartbeat,
    ah.cpu_usage,
    ah.memory_usage,
        CASE
            WHEN (ah.last_heartbeat > (CURRENT_TIMESTAMP - '00:05:00'::interval)) THEN 'HEALTHY'::text
            WHEN (ah.last_heartbeat > (CURRENT_TIMESTAMP - '00:30:00'::interval)) THEN 'STALE'::text
            ELSE 'OFFLINE'::text
        END AS connectivity_status
   FROM (public.agents a
     LEFT JOIN public.agent_health ah ON ((a.id = ah.agent_id)));


ALTER VIEW public.agent_status_overview OWNER TO sutazai;

--
-- TOC entry 221 (class 1259 OID 24590)
-- Name: agents_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.agents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.agents_id_seq OWNER TO sutazai;

--
-- TOC entry 3907 (class 0 OID 0)
-- Dependencies: 221
-- Name: agents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agents_id_seq OWNED BY public.agents.id;


--
-- TOC entry 244 (class 1259 OID 24808)
-- Name: api_usage_logs; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.api_usage_logs (
    id integer NOT NULL,
    endpoint character varying(255) NOT NULL,
    method character varying(10) NOT NULL,
    user_id integer,
    agent_id integer,
    response_code integer,
    response_time numeric(8,3),
    request_size integer,
    response_size integer,
    ip_address inet,
    user_agent text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.api_usage_logs OWNER TO sutazai;

--
-- TOC entry 243 (class 1259 OID 24807)
-- Name: api_usage_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.api_usage_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.api_usage_logs_id_seq OWNER TO sutazai;

--
-- TOC entry 3908 (class 0 OID 0)
-- Dependencies: 243
-- Name: api_usage_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.api_usage_logs_id_seq OWNED BY public.api_usage_logs.id;


--
-- TOC entry 226 (class 1259 OID 24628)
-- Name: chat_history; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.chat_history (
    id integer NOT NULL,
    user_id integer,
    message text NOT NULL,
    response text,
    agent_used character varying(100),
    tokens_used integer,
    response_time double precision,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.chat_history OWNER TO sutazai;

--
-- TOC entry 225 (class 1259 OID 24627)
-- Name: chat_history_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.chat_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.chat_history_id_seq OWNER TO sutazai;

--
-- TOC entry 3909 (class 0 OID 0)
-- Dependencies: 225
-- Name: chat_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.chat_history_id_seq OWNED BY public.chat_history.id;


--
-- TOC entry 251 (class 1259 OID 25425)
-- Name: connection_stats; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.connection_stats AS
 SELECT datname,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    EXTRACT(epoch FROM (now() - query_start)) AS query_duration_seconds
   FROM pg_stat_activity
  WHERE (datname = 'sutazai'::name)
  ORDER BY query_start DESC;


ALTER VIEW public.connection_stats OWNER TO sutazai;

--
-- TOC entry 252 (class 1259 OID 25429)
-- Name: connection_summary; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.connection_summary AS
 SELECT state,
    count(*) AS connection_count,
    avg(EXTRACT(epoch FROM (now() - state_change))) AS avg_state_duration
   FROM pg_stat_activity
  WHERE (datname = 'sutazai'::name)
  GROUP BY state
  ORDER BY (count(*)) DESC;


ALTER VIEW public.connection_summary OWNER TO sutazai;

--
-- TOC entry 253 (class 1259 OID 25433)
-- Name: db_activity_monitor; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.db_activity_monitor AS
 SELECT 'Total Connections'::text AS metric,
    (count(*))::text AS value
   FROM pg_stat_activity
  WHERE (pg_stat_activity.datname = 'sutazai'::name)
UNION ALL
 SELECT 'Active Queries'::text AS metric,
    (count(*))::text AS value
   FROM pg_stat_activity
  WHERE ((pg_stat_activity.datname = 'sutazai'::name) AND (pg_stat_activity.state = 'active'::text))
UNION ALL
 SELECT 'Idle Connections'::text AS metric,
    (count(*))::text AS value
   FROM pg_stat_activity
  WHERE ((pg_stat_activity.datname = 'sutazai'::name) AND (pg_stat_activity.state = 'idle'::text))
UNION ALL
 SELECT 'Long Running Queries'::text AS metric,
    (count(*))::text AS value
   FROM pg_stat_activity
  WHERE ((pg_stat_activity.datname = 'sutazai'::name) AND (pg_stat_activity.state = 'active'::text) AND (pg_stat_activity.query_start < (now() - '00:00:30'::interval)));


ALTER VIEW public.db_activity_monitor OWNER TO sutazai;

--
-- TOC entry 240 (class 1259 OID 24770)
-- Name: knowledge_documents; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.knowledge_documents (
    id integer NOT NULL,
    title character varying(500),
    content_preview text,
    full_content text,
    document_type character varying(100),
    source_path character varying(1000),
    collection_id integer,
    embedding_status character varying(50) DEFAULT 'pending'::character varying,
    processed_at timestamp without time zone,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.knowledge_documents OWNER TO sutazai;

--
-- TOC entry 239 (class 1259 OID 24769)
-- Name: knowledge_documents_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.knowledge_documents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.knowledge_documents_id_seq OWNER TO sutazai;

--
-- TOC entry 3910 (class 0 OID 0)
-- Dependencies: 239
-- Name: knowledge_documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.knowledge_documents_id_seq OWNED BY public.knowledge_documents.id;


--
-- TOC entry 236 (class 1259 OID 24737)
-- Name: model_registry; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.model_registry (
    id integer NOT NULL,
    model_name character varying(255) NOT NULL,
    model_type character varying(100) NOT NULL,
    size_mb numeric(10,2),
    status character varying(50) DEFAULT 'available'::character varying,
    ollama_status character varying(50),
    usage_count integer DEFAULT 0,
    last_used timestamp without time zone,
    file_path text,
    parameters jsonb DEFAULT '{}'::jsonb,
    capabilities jsonb DEFAULT '[]'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.model_registry OWNER TO sutazai;

--
-- TOC entry 235 (class 1259 OID 24736)
-- Name: model_registry_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.model_registry_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.model_registry_id_seq OWNER TO sutazai;

--
-- TOC entry 3911 (class 0 OID 0)
-- Dependencies: 235
-- Name: model_registry_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.model_registry_id_seq OWNED BY public.model_registry.id;


--
-- TOC entry 242 (class 1259 OID 24791)
-- Name: orchestration_sessions; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.orchestration_sessions (
    id integer NOT NULL,
    session_name character varying(255),
    task_description text NOT NULL,
    agents_involved jsonb DEFAULT '[]'::jsonb NOT NULL,
    strategy character varying(50) DEFAULT 'collaborative'::character varying,
    status character varying(50) DEFAULT 'pending'::character varying,
    progress numeric(5,2) DEFAULT 0.00,
    result jsonb,
    error_message text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    started_at timestamp without time zone,
    completed_at timestamp without time zone,
    metadata jsonb DEFAULT '{}'::jsonb
);


ALTER TABLE public.orchestration_sessions OWNER TO sutazai;

--
-- TOC entry 241 (class 1259 OID 24790)
-- Name: orchestration_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.orchestration_sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.orchestration_sessions_id_seq OWNER TO sutazai;

--
-- TOC entry 3912 (class 0 OID 0)
-- Dependencies: 241
-- Name: orchestration_sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.orchestration_sessions_id_seq OWNED BY public.orchestration_sessions.id;


--
-- TOC entry 250 (class 1259 OID 24873)
-- Name: performance_metrics; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.performance_metrics AS
 SELECT 'agent_response_time'::text AS metric_name,
    avg(ah.response_time) AS avg_value,
    max(ah.response_time) AS max_value,
    min(ah.response_time) AS min_value,
    count(*) AS sample_count
   FROM public.agent_health ah
  WHERE (ah.last_heartbeat > (CURRENT_TIMESTAMP - '01:00:00'::interval))
  GROUP BY 'agent_response_time'::text
UNION ALL
 SELECT 'api_response_time'::text AS metric_name,
    avg(aul.response_time) AS avg_value,
    max(aul.response_time) AS max_value,
    min(aul.response_time) AS min_value,
    count(*) AS sample_count
   FROM public.api_usage_logs aul
  WHERE (aul.created_at > (CURRENT_TIMESTAMP - '01:00:00'::interval))
  GROUP BY 'api_response_time'::text;


ALTER VIEW public.performance_metrics OWNER TO sutazai;

--
-- TOC entry 224 (class 1259 OID 24605)
-- Name: tasks; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.tasks (
    id integer NOT NULL,
    title character varying(255) NOT NULL,
    description text,
    agent_id integer,
    user_id integer,
    status character varying(50) DEFAULT 'pending'::character varying,
    priority integer DEFAULT 5,
    payload jsonb DEFAULT '{}'::jsonb,
    result jsonb,
    error_message text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    started_at timestamp without time zone,
    completed_at timestamp without time zone
);


ALTER TABLE public.tasks OWNER TO sutazai;

--
-- TOC entry 220 (class 1259 OID 24577)
-- Name: users; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    password_hash character varying(255) NOT NULL,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.users OWNER TO sutazai;

--
-- TOC entry 249 (class 1259 OID 24868)
-- Name: recent_activity; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.recent_activity AS
 SELECT 'task'::text AS activity_type,
    t.title AS description,
    t.status,
    t.created_at,
    u.username AS initiated_by,
    a.name AS agent_involved
   FROM ((public.tasks t
     LEFT JOIN public.users u ON ((t.user_id = u.id)))
     LEFT JOIN public.agents a ON ((t.agent_id = a.id)))
  WHERE (t.created_at > (CURRENT_TIMESTAMP - '24:00:00'::interval))
UNION ALL
 SELECT 'orchestration'::text AS activity_type,
    os.task_description AS description,
    os.status,
    os.created_at,
    'system'::character varying AS initiated_by,
    NULL::character varying AS agent_involved
   FROM public.orchestration_sessions os
  WHERE (os.created_at > (CURRENT_TIMESTAMP - '24:00:00'::interval))
  ORDER BY 4 DESC
 LIMIT 50;


ALTER VIEW public.recent_activity OWNER TO sutazai;

--
-- TOC entry 232 (class 1259 OID 24686)
-- Name: sessions; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.sessions (
    id integer NOT NULL,
    user_id integer,
    token character varying(255) NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    is_active boolean DEFAULT true,
    user_agent text,
    ip_address inet,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_accessed timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.sessions OWNER TO sutazai;

--
-- TOC entry 231 (class 1259 OID 24685)
-- Name: sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sessions_id_seq OWNER TO sutazai;

--
-- TOC entry 3913 (class 0 OID 0)
-- Dependencies: 231
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
-- TOC entry 246 (class 1259 OID 24831)
-- Name: system_alerts; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.system_alerts (
    id integer NOT NULL,
    alert_type character varying(100) NOT NULL,
    severity character varying(20) DEFAULT 'medium'::character varying NOT NULL,
    title character varying(255) NOT NULL,
    description text,
    source character varying(100),
    status character varying(50) DEFAULT 'active'::character varying,
    resolved_at timestamp without time zone,
    resolved_by integer,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.system_alerts OWNER TO sutazai;

--
-- TOC entry 245 (class 1259 OID 24830)
-- Name: system_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.system_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.system_alerts_id_seq OWNER TO sutazai;

--
-- TOC entry 3914 (class 0 OID 0)
-- Dependencies: 245
-- Name: system_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.system_alerts_id_seq OWNED BY public.system_alerts.id;


--
-- TOC entry 248 (class 1259 OID 24863)
-- Name: system_health_dashboard; Type: VIEW; Schema: public; Owner: sutazai
--

CREATE VIEW public.system_health_dashboard AS
 SELECT ( SELECT count(*) AS count
           FROM public.agents
          WHERE (agents.is_active = true)) AS total_agents,
    ( SELECT count(*) AS count
           FROM public.agent_health
          WHERE (((agent_health.status)::text = 'healthy'::text) AND (agent_health.last_heartbeat > (CURRENT_TIMESTAMP - '00:05:00'::interval)))) AS healthy_agents,
    ( SELECT count(*) AS count
           FROM public.tasks
          WHERE ((tasks.status)::text = 'pending'::text)) AS pending_tasks,
    ( SELECT count(*) AS count
           FROM public.tasks
          WHERE ((tasks.status)::text = 'running'::text)) AS running_tasks,
    ( SELECT count(*) AS count
           FROM public.orchestration_sessions
          WHERE ((orchestration_sessions.status)::text = 'active'::text)) AS active_orchestrations,
    ( SELECT count(*) AS count
           FROM public.system_alerts
          WHERE ((system_alerts.status)::text = 'active'::text)) AS active_alerts,
    ( SELECT count(*) AS count
           FROM public.model_registry
          WHERE ((model_registry.status)::text = 'active'::text)) AS active_models,
    CURRENT_TIMESTAMP AS snapshot_time;


ALTER VIEW public.system_health_dashboard OWNER TO sutazai;

--
-- TOC entry 230 (class 1259 OID 24663)
-- Name: system_metrics; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.system_metrics (
    id integer NOT NULL,
    metric_name character varying(100) NOT NULL,
    metric_value double precision NOT NULL,
    tags jsonb DEFAULT '{}'::jsonb,
    recorded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.system_metrics OWNER TO sutazai;

--
-- TOC entry 229 (class 1259 OID 24662)
-- Name: system_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.system_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.system_metrics_id_seq OWNER TO sutazai;

--
-- TOC entry 3915 (class 0 OID 0)
-- Dependencies: 229
-- Name: system_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.system_metrics_id_seq OWNED BY public.system_metrics.id;


--
-- TOC entry 223 (class 1259 OID 24604)
-- Name: tasks_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.tasks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.tasks_id_seq OWNER TO sutazai;

--
-- TOC entry 3916 (class 0 OID 0)
-- Dependencies: 223
-- Name: tasks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.tasks_id_seq OWNED BY public.tasks.id;


--
-- TOC entry 219 (class 1259 OID 24576)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO sutazai;

--
-- TOC entry 3917 (class 0 OID 0)
-- Dependencies: 219
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 238 (class 1259 OID 24757)
-- Name: vector_collections; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.vector_collections (
    id integer NOT NULL,
    collection_name character varying(255) NOT NULL,
    database_type character varying(50) NOT NULL,
    dimension integer NOT NULL,
    document_count integer DEFAULT 0,
    status character varying(50) DEFAULT 'active'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.vector_collections OWNER TO sutazai;

--
-- TOC entry 237 (class 1259 OID 24756)
-- Name: vector_collections_id_seq; Type: SEQUENCE; Schema: public; Owner: sutazai
--

CREATE SEQUENCE public.vector_collections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.vector_collections_id_seq OWNER TO sutazai;

--
-- TOC entry 3918 (class 0 OID 0)
-- Dependencies: 237
-- Name: vector_collections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.vector_collections_id_seq OWNED BY public.vector_collections.id;


--
-- TOC entry 3571 (class 2604 OID 24646)
-- Name: agent_executions id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions ALTER COLUMN id SET DEFAULT nextval('public.agent_executions_id_seq'::regclass);


--
-- TOC entry 3580 (class 2604 OID 24712)
-- Name: agent_health id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health ALTER COLUMN id SET DEFAULT nextval('public.agent_health_id_seq'::regclass);


--
-- TOC entry 3560 (class 2604 OID 24594)
-- Name: agents id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents ALTER COLUMN id SET DEFAULT nextval('public.agents_id_seq'::regclass);


--
-- TOC entry 3616 (class 2604 OID 24811)
-- Name: api_usage_logs id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.api_usage_logs ALTER COLUMN id SET DEFAULT nextval('public.api_usage_logs_id_seq'::regclass);


--
-- TOC entry 3569 (class 2604 OID 24631)
-- Name: chat_history id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history ALTER COLUMN id SET DEFAULT nextval('public.chat_history_id_seq'::regclass);


--
-- TOC entry 3604 (class 2604 OID 24773)
-- Name: knowledge_documents id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.knowledge_documents ALTER COLUMN id SET DEFAULT nextval('public.knowledge_documents_id_seq'::regclass);


--
-- TOC entry 3592 (class 2604 OID 24740)
-- Name: model_registry id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry ALTER COLUMN id SET DEFAULT nextval('public.model_registry_id_seq'::regclass);


--
-- TOC entry 3609 (class 2604 OID 24794)
-- Name: orchestration_sessions id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.orchestration_sessions ALTER COLUMN id SET DEFAULT nextval('public.orchestration_sessions_id_seq'::regclass);


--
-- TOC entry 3576 (class 2604 OID 24689)
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- TOC entry 3618 (class 2604 OID 24834)
-- Name: system_alerts id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts ALTER COLUMN id SET DEFAULT nextval('public.system_alerts_id_seq'::regclass);


--
-- TOC entry 3573 (class 2604 OID 24666)
-- Name: system_metrics id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_metrics ALTER COLUMN id SET DEFAULT nextval('public.system_metrics_id_seq'::regclass);


--
-- TOC entry 3564 (class 2604 OID 24608)
-- Name: tasks id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks ALTER COLUMN id SET DEFAULT nextval('public.tasks_id_seq'::regclass);


--
-- TOC entry 3556 (class 2604 OID 24580)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 3599 (class 2604 OID 24760)
-- Name: vector_collections id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.vector_collections ALTER COLUMN id SET DEFAULT nextval('public.vector_collections_id_seq'::regclass);


--
-- TOC entry 3877 (class 0 OID 24643)
-- Dependencies: 228
-- Data for Name: agent_executions; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agent_executions (id, agent_id, task_id, status, input_data, output_data, execution_time, error_message, created_at) FROM stdin;
\.


--
-- TOC entry 3883 (class 0 OID 24709)
-- Dependencies: 234
-- Data for Name: agent_health; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agent_health (id, agent_id, status, last_heartbeat, cpu_usage, memory_usage, disk_usage, response_time, error_count, success_count, metadata, created_at, updated_at) FROM stdin;
\.


--
-- TOC entry 3871 (class 0 OID 24591)
-- Dependencies: 222
-- Data for Name: agents; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agents (id, name, type, description, endpoint, port, is_active, capabilities, created_at) FROM stdin;
1	health-monitor	monitoring	\N	http://health-monitor:8080	10210	t	["health_check", "metrics"]	2025-08-06 12:07:04.291468
2	task-coordinator	orchestration	\N	http://task-coordinator:8080	10450	t	["task_routing", "scheduling"]	2025-08-06 12:07:04.291468
3	ollama-service	llm	\N	http://ollama:11434	11434	t	["text_generation", "chat"]	2025-08-06 12:07:04.291468
5	ai-agent-orchestrator	orchestration	Main AI agent orchestrator	http://ai-agent-orchestrator:8589	8589	t	["orchestration", "coordination", "task_routing"]	2025-08-06 14:21:19.946343
6	multi-agent-coordinator	coordination	Multi-agent system coordinator	http://multi-agent-coordinator:8587	8587	t	["coordination", "scheduling", "resource_management"]	2025-08-06 14:21:19.946343
7	resource-arbitration-agent	resource	Resource allocation and arbitration	http://resource-arbitration-agent:8588	8588	t	["resource_allocation", "load_balancing"]	2025-08-06 14:21:19.946343
8	task-assignment-coordinator	task	Task assignment and routing	http://task-assignment-coordinator:8551	8551	t	["task_assignment", "routing", "scheduling"]	2025-08-06 14:21:19.946343
9	hardware-resource-optimizer	optimization	Hardware resource optimization	http://hardware-resource-optimizer:8002	8002	t	["hardware_optimization", "performance_tuning"]	2025-08-06 14:21:19.946343
10	ollama-integration-specialist	integration	Ollama LLM integration specialist	http://ollama-integration-specialist:11015	11015	t	["llm_integration", "model_management"]	2025-08-06 14:21:19.946343
11	ai-metrics-exporter	monitoring	AI system metrics collection	http://ai-metrics-exporter:11063	11063	f	["metrics", "monitoring", "telemetry"]	2025-08-06 14:21:19.946343
\.


--
-- TOC entry 3893 (class 0 OID 24808)
-- Dependencies: 244
-- Data for Name: api_usage_logs; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.api_usage_logs (id, endpoint, method, user_id, agent_id, response_code, response_time, request_size, response_size, ip_address, user_agent, created_at) FROM stdin;
\.


--
-- TOC entry 3875 (class 0 OID 24628)
-- Dependencies: 226
-- Data for Name: chat_history; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.chat_history (id, user_id, message, response, agent_used, tokens_used, response_time, created_at) FROM stdin;
\.


--
-- TOC entry 3889 (class 0 OID 24770)
-- Dependencies: 240
-- Data for Name: knowledge_documents; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.knowledge_documents (id, title, content_preview, full_content, document_type, source_path, collection_id, embedding_status, processed_at, metadata, created_at, updated_at) FROM stdin;
\.


--
-- TOC entry 3885 (class 0 OID 24737)
-- Dependencies: 236
-- Data for Name: model_registry; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.model_registry (id, model_name, model_type, size_mb, status, ollama_status, usage_count, last_used, file_path, parameters, capabilities, created_at, updated_at) FROM stdin;
1	tinyllama	llm	637.00	active	loaded	0	\N	\N	{"parameters": "1.1B", "quantization": "Q4_0", "context_length": 2048}	[]	2025-08-06 14:21:19.950298	2025-08-06 14:21:19.950298
2	gpt-oss	llm	2500.00	available	not_loaded	0	\N	\N	{"description": "Open source GPT variant", "context_length": 4096}	[]	2025-08-06 14:21:19.950298	2025-08-06 14:21:19.950298
\.


--
-- TOC entry 3891 (class 0 OID 24791)
-- Dependencies: 242
-- Data for Name: orchestration_sessions; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.orchestration_sessions (id, session_name, task_description, agents_involved, strategy, status, progress, result, error_message, created_at, started_at, completed_at, metadata) FROM stdin;
\.


--
-- TOC entry 3881 (class 0 OID 24686)
-- Dependencies: 232
-- Data for Name: sessions; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.sessions (id, user_id, token, expires_at, is_active, user_agent, ip_address, created_at, last_accessed) FROM stdin;
\.


--
-- TOC entry 3895 (class 0 OID 24831)
-- Dependencies: 246
-- Data for Name: system_alerts; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.system_alerts (id, alert_type, severity, title, description, source, status, resolved_at, resolved_by, metadata, created_at) FROM stdin;
\.


--
-- TOC entry 3879 (class 0 OID 24663)
-- Dependencies: 230
-- Data for Name: system_metrics; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.system_metrics (id, metric_name, metric_value, tags, recorded_at) FROM stdin;
\.


--
-- TOC entry 3873 (class 0 OID 24605)
-- Dependencies: 224
-- Data for Name: tasks; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.tasks (id, title, description, agent_id, user_id, status, priority, payload, result, error_message, created_at, started_at, completed_at) FROM stdin;
\.


--
-- TOC entry 3869 (class 0 OID 24577)
-- Dependencies: 220
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.users (id, username, email, password_hash, is_active, created_at, updated_at) FROM stdin;
1	admin	admin@sutazai.local	$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewjyJyM7QK8kL5yC	t	2025-08-06 14:21:19.941818	2025-08-06 14:21:19.941818
2	system	system@sutazai.local	$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewjyJyM7QK8kL5yC	t	2025-08-06 14:21:19.941818	2025-08-06 14:21:19.941818
\.


--
-- TOC entry 3887 (class 0 OID 24757)
-- Dependencies: 238
-- Data for Name: vector_collections; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.vector_collections (id, collection_name, database_type, dimension, document_count, status, created_at, updated_at) FROM stdin;
1	default_qdrant	qdrant	384	0	active	2025-08-06 14:21:19.953882	2025-08-06 14:21:19.953882
2	default_faiss	faiss	384	0	active	2025-08-06 14:21:19.953882	2025-08-06 14:21:19.953882
3	default_chromadb	chromadb	384	0	degraded	2025-08-06 14:21:19.953882	2025-08-06 14:21:19.953882
\.


--
-- TOC entry 3919 (class 0 OID 0)
-- Dependencies: 227
-- Name: agent_executions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agent_executions_id_seq', 1, false);


--
-- TOC entry 3920 (class 0 OID 0)
-- Dependencies: 233
-- Name: agent_health_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agent_health_id_seq', 1, false);


--
-- TOC entry 3921 (class 0 OID 0)
-- Dependencies: 221
-- Name: agents_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agents_id_seq', 11, true);


--
-- TOC entry 3922 (class 0 OID 0)
-- Dependencies: 243
-- Name: api_usage_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.api_usage_logs_id_seq', 1, false);


--
-- TOC entry 3923 (class 0 OID 0)
-- Dependencies: 225
-- Name: chat_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.chat_history_id_seq', 1, false);


--
-- TOC entry 3924 (class 0 OID 0)
-- Dependencies: 239
-- Name: knowledge_documents_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.knowledge_documents_id_seq', 1, false);


--
-- TOC entry 3925 (class 0 OID 0)
-- Dependencies: 235
-- Name: model_registry_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.model_registry_id_seq', 2, true);


--
-- TOC entry 3926 (class 0 OID 0)
-- Dependencies: 241
-- Name: orchestration_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.orchestration_sessions_id_seq', 1, false);


--
-- TOC entry 3927 (class 0 OID 0)
-- Dependencies: 231
-- Name: sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.sessions_id_seq', 1, false);


--
-- TOC entry 3928 (class 0 OID 0)
-- Dependencies: 245
-- Name: system_alerts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.system_alerts_id_seq', 1, false);


--
-- TOC entry 3929 (class 0 OID 0)
-- Dependencies: 229
-- Name: system_metrics_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.system_metrics_id_seq', 1, true);


--
-- TOC entry 3930 (class 0 OID 0)
-- Dependencies: 223
-- Name: tasks_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.tasks_id_seq', 1, false);


--
-- TOC entry 3931 (class 0 OID 0)
-- Dependencies: 219
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.users_id_seq', 2, true);


--
-- TOC entry 3932 (class 0 OID 0)
-- Dependencies: 237
-- Name: vector_collections_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.vector_collections_id_seq', 3, true);


--
-- TOC entry 3649 (class 2606 OID 24651)
-- Name: agent_executions agent_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);


--
-- TOC entry 3665 (class 2606 OID 24727)
-- Name: agent_health agent_health_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health
    ADD CONSTRAINT agent_health_pkey PRIMARY KEY (id);


--
-- TOC entry 3631 (class 2606 OID 24603)
-- Name: agents agents_name_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_name_key UNIQUE (name);


--
-- TOC entry 3633 (class 2606 OID 24601)
-- Name: agents agents_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_pkey PRIMARY KEY (id);


--
-- TOC entry 3692 (class 2606 OID 24816)
-- Name: api_usage_logs api_usage_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.api_usage_logs
    ADD CONSTRAINT api_usage_logs_pkey PRIMARY KEY (id);


--
-- TOC entry 3645 (class 2606 OID 24636)
-- Name: chat_history chat_history_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history
    ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);


--
-- TOC entry 3686 (class 2606 OID 24781)
-- Name: knowledge_documents knowledge_documents_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.knowledge_documents
    ADD CONSTRAINT knowledge_documents_pkey PRIMARY KEY (id);


--
-- TOC entry 3674 (class 2606 OID 24752)
-- Name: model_registry model_registry_model_name_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry
    ADD CONSTRAINT model_registry_model_name_key UNIQUE (model_name);


--
-- TOC entry 3676 (class 2606 OID 24750)
-- Name: model_registry model_registry_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry
    ADD CONSTRAINT model_registry_pkey PRIMARY KEY (id);


--
-- TOC entry 3690 (class 2606 OID 24804)
-- Name: orchestration_sessions orchestration_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.orchestration_sessions
    ADD CONSTRAINT orchestration_sessions_pkey PRIMARY KEY (id);


--
-- TOC entry 3661 (class 2606 OID 24696)
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- TOC entry 3663 (class 2606 OID 24698)
-- Name: sessions sessions_token_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_token_key UNIQUE (token);


--
-- TOC entry 3701 (class 2606 OID 24842)
-- Name: system_alerts system_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts
    ADD CONSTRAINT system_alerts_pkey PRIMARY KEY (id);


--
-- TOC entry 3655 (class 2606 OID 24672)
-- Name: system_metrics system_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_metrics
    ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);


--
-- TOC entry 3643 (class 2606 OID 24616)
-- Name: tasks tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);


--
-- TOC entry 3625 (class 2606 OID 24589)
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- TOC entry 3627 (class 2606 OID 24585)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 3629 (class 2606 OID 24587)
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- TOC entry 3678 (class 2606 OID 24768)
-- Name: vector_collections vector_collections_collection_name_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.vector_collections
    ADD CONSTRAINT vector_collections_collection_name_key UNIQUE (collection_name);


--
-- TOC entry 3680 (class 2606 OID 24766)
-- Name: vector_collections vector_collections_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.vector_collections
    ADD CONSTRAINT vector_collections_pkey PRIMARY KEY (id);


--
-- TOC entry 3650 (class 1259 OID 24676)
-- Name: idx_agent_executions_agent_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_executions_agent_id ON public.agent_executions USING btree (agent_id);


--
-- TOC entry 3666 (class 1259 OID 24733)
-- Name: idx_agent_health_agent_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_agent_id ON public.agent_health USING btree (agent_id);


--
-- TOC entry 3667 (class 1259 OID 25419)
-- Name: idx_agent_health_agent_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_agent_status ON public.agent_health USING btree (agent_id, status, last_heartbeat DESC);


--
-- TOC entry 3668 (class 1259 OID 24735)
-- Name: idx_agent_health_heartbeat; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_heartbeat ON public.agent_health USING btree (last_heartbeat);


--
-- TOC entry 3669 (class 1259 OID 24734)
-- Name: idx_agent_health_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_status ON public.agent_health USING btree (status);


--
-- TOC entry 3634 (class 1259 OID 25421)
-- Name: idx_agents_active; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agents_active ON public.agents USING btree (name) WHERE (is_active = true);


--
-- TOC entry 3635 (class 1259 OID 25414)
-- Name: idx_agents_capabilities_gin; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agents_capabilities_gin ON public.agents USING gin (capabilities);


--
-- TOC entry 3693 (class 1259 OID 24828)
-- Name: idx_api_usage_created; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_api_usage_created ON public.api_usage_logs USING btree (created_at);


--
-- TOC entry 3694 (class 1259 OID 24827)
-- Name: idx_api_usage_endpoint; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_api_usage_endpoint ON public.api_usage_logs USING btree (endpoint);


--
-- TOC entry 3695 (class 1259 OID 24829)
-- Name: idx_api_usage_user; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_api_usage_user ON public.api_usage_logs USING btree (user_id);


--
-- TOC entry 3646 (class 1259 OID 25420)
-- Name: idx_chat_history_user_created; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_chat_history_user_created ON public.chat_history USING btree (user_id, created_at DESC);


--
-- TOC entry 3647 (class 1259 OID 24675)
-- Name: idx_chat_history_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_chat_history_user_id ON public.chat_history USING btree (user_id);


--
-- TOC entry 3681 (class 1259 OID 24787)
-- Name: idx_knowledge_documents_collection; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_knowledge_documents_collection ON public.knowledge_documents USING btree (collection_id);


--
-- TOC entry 3682 (class 1259 OID 24789)
-- Name: idx_knowledge_documents_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_knowledge_documents_status ON public.knowledge_documents USING btree (embedding_status);


--
-- TOC entry 3683 (class 1259 OID 25424)
-- Name: idx_knowledge_documents_title_trgm; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_knowledge_documents_title_trgm ON public.knowledge_documents USING gin (title public.gin_trgm_ops);


--
-- TOC entry 3684 (class 1259 OID 24788)
-- Name: idx_knowledge_documents_type; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_knowledge_documents_type ON public.knowledge_documents USING btree (document_type);


--
-- TOC entry 3670 (class 1259 OID 24753)
-- Name: idx_model_registry_name; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_model_registry_name ON public.model_registry USING btree (model_name);


--
-- TOC entry 3671 (class 1259 OID 24755)
-- Name: idx_model_registry_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_model_registry_status ON public.model_registry USING btree (status);


--
-- TOC entry 3672 (class 1259 OID 24754)
-- Name: idx_model_registry_type; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_model_registry_type ON public.model_registry USING btree (model_type);


--
-- TOC entry 3687 (class 1259 OID 24806)
-- Name: idx_orchestration_created; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_orchestration_created ON public.orchestration_sessions USING btree (created_at);


--
-- TOC entry 3688 (class 1259 OID 24805)
-- Name: idx_orchestration_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_orchestration_status ON public.orchestration_sessions USING btree (status);


--
-- TOC entry 3656 (class 1259 OID 24707)
-- Name: idx_sessions_active; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_active ON public.sessions USING btree (is_active);


--
-- TOC entry 3657 (class 1259 OID 24706)
-- Name: idx_sessions_expires_at; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_expires_at ON public.sessions USING btree (expires_at);


--
-- TOC entry 3658 (class 1259 OID 24705)
-- Name: idx_sessions_token; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_token ON public.sessions USING btree (token);


--
-- TOC entry 3659 (class 1259 OID 24704)
-- Name: idx_sessions_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_user_id ON public.sessions USING btree (user_id);


--
-- TOC entry 3696 (class 1259 OID 24851)
-- Name: idx_system_alerts_created; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_alerts_created ON public.system_alerts USING btree (created_at);


--
-- TOC entry 3697 (class 1259 OID 24849)
-- Name: idx_system_alerts_severity; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_alerts_severity ON public.system_alerts USING btree (severity);


--
-- TOC entry 3698 (class 1259 OID 24850)
-- Name: idx_system_alerts_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_alerts_status ON public.system_alerts USING btree (status);


--
-- TOC entry 3699 (class 1259 OID 24848)
-- Name: idx_system_alerts_type; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_alerts_type ON public.system_alerts USING btree (alert_type);


--
-- TOC entry 3651 (class 1259 OID 24677)
-- Name: idx_system_metrics_name; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_metrics_name ON public.system_metrics USING btree (metric_name);


--
-- TOC entry 3652 (class 1259 OID 24678)
-- Name: idx_system_metrics_recorded_at; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_metrics_recorded_at ON public.system_metrics USING btree (recorded_at);


--
-- TOC entry 3653 (class 1259 OID 25417)
-- Name: idx_system_metrics_tags_gin; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_metrics_tags_gin ON public.system_metrics USING gin (tags);


--
-- TOC entry 3636 (class 1259 OID 25415)
-- Name: idx_tasks_payload_gin; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_payload_gin ON public.tasks USING gin (payload);


--
-- TOC entry 3637 (class 1259 OID 25416)
-- Name: idx_tasks_result_gin; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_result_gin ON public.tasks USING gin (result);


--
-- TOC entry 3638 (class 1259 OID 24673)
-- Name: idx_tasks_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_status ON public.tasks USING btree (status);


--
-- TOC entry 3639 (class 1259 OID 25418)
-- Name: idx_tasks_status_created_at; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_status_created_at ON public.tasks USING btree (status, created_at DESC);


--
-- TOC entry 3640 (class 1259 OID 25423)
-- Name: idx_tasks_title_trgm; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_title_trgm ON public.tasks USING gin (title public.gin_trgm_ops);


--
-- TOC entry 3641 (class 1259 OID 24674)
-- Name: idx_tasks_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_user_id ON public.tasks USING btree (user_id);


--
-- TOC entry 3623 (class 1259 OID 25422)
-- Name: idx_users_active; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_users_active ON public.users USING btree (username) WHERE (is_active = true);


--
-- TOC entry 3714 (class 2620 OID 24854)
-- Name: agent_health update_agent_health_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_agent_health_updated_at BEFORE UPDATE ON public.agent_health FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 3717 (class 2620 OID 24857)
-- Name: knowledge_documents update_knowledge_documents_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_knowledge_documents_updated_at BEFORE UPDATE ON public.knowledge_documents FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 3715 (class 2620 OID 24855)
-- Name: model_registry update_model_registry_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 3713 (class 2620 OID 24853)
-- Name: sessions update_sessions_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON public.sessions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 3716 (class 2620 OID 24856)
-- Name: vector_collections update_vector_collections_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_vector_collections_updated_at BEFORE UPDATE ON public.vector_collections FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 3705 (class 2606 OID 24652)
-- Name: agent_executions agent_executions_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id);


--
-- TOC entry 3706 (class 2606 OID 24657)
-- Name: agent_executions agent_executions_task_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.tasks(id);


--
-- TOC entry 3708 (class 2606 OID 24728)
-- Name: agent_health agent_health_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health
    ADD CONSTRAINT agent_health_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id) ON DELETE CASCADE;


--
-- TOC entry 3710 (class 2606 OID 24822)
-- Name: api_usage_logs api_usage_logs_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.api_usage_logs
    ADD CONSTRAINT api_usage_logs_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id);


--
-- TOC entry 3711 (class 2606 OID 24817)
-- Name: api_usage_logs api_usage_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.api_usage_logs
    ADD CONSTRAINT api_usage_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- TOC entry 3704 (class 2606 OID 24637)
-- Name: chat_history chat_history_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history
    ADD CONSTRAINT chat_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- TOC entry 3709 (class 2606 OID 24782)
-- Name: knowledge_documents knowledge_documents_collection_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.knowledge_documents
    ADD CONSTRAINT knowledge_documents_collection_id_fkey FOREIGN KEY (collection_id) REFERENCES public.vector_collections(id);


--
-- TOC entry 3707 (class 2606 OID 24699)
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 3712 (class 2606 OID 24843)
-- Name: system_alerts system_alerts_resolved_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts
    ADD CONSTRAINT system_alerts_resolved_by_fkey FOREIGN KEY (resolved_by) REFERENCES public.users(id);


--
-- TOC entry 3702 (class 2606 OID 24617)
-- Name: tasks tasks_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id);


--
-- TOC entry 3703 (class 2606 OID 24622)
-- Name: tasks tasks_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


-- Completed on 2025-08-06 14:24:39 UTC

--
-- PostgreSQL database dump complete
--

