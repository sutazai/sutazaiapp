--
-- PostgreSQL database cluster dump
--

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Roles
--

CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:0PN8xCpLElZgMLLDhDSb5w==$4XM5KUDZoIK/XmUlJtoFjcsF0OEI++2IoKuZc0o14X4=:iUhMZnUu0RqBRJVO9QuA/YC7ozimAbUrF1Kg0hsDeOQ=';
CREATE ROLE sutazai;
ALTER ROLE sutazai WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION BYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:mnXNG6D5OaGhJp9z3LSMMA==$Ydnh+zZLsqYkgERshiniRI4xhGZMQKBkrERhEzPOZCA=:XesCr6m/7FnDjHEPnMPqmJB29V4CBnbGn/IMR0Tezlc=';

--
-- User Configurations
--








--
-- Databases
--

--
-- Database "template1" dump
--

\connect template1

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

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
-- PostgreSQL database dump complete
--

--
-- Database "postgres" dump
--

\connect postgres

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

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
-- PostgreSQL database dump complete
--

--
-- Database "sutazai" dump
--

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

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
-- Name: sutazai; Type: DATABASE; Schema: -; Owner: sutazai
--

CREATE DATABASE sutazai WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE sutazai OWNER TO sutazai;

\connect sutazai

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
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
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
-- Name: agent_executions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agent_executions_id_seq OWNED BY public.agent_executions.id;


--
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
-- Name: agent_health_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agent_health_id_seq OWNED BY public.agent_health.id;


--
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
-- Name: agents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.agents_id_seq OWNED BY public.agents.id;


--
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
-- Name: chat_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.chat_history_id_seq OWNED BY public.chat_history.id;


--
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
-- Name: model_registry_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.model_registry_id_seq OWNED BY public.model_registry.id;


--
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
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
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
-- Name: system_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.system_alerts_id_seq OWNED BY public.system_alerts.id;


--
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
-- Name: system_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.system_metrics_id_seq OWNED BY public.system_metrics.id;


--
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
-- Name: tasks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.tasks_id_seq OWNED BY public.tasks.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: sutazai
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    password_hash character varying(255) NOT NULL,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    is_admin boolean DEFAULT false,
    last_login timestamp with time zone,
    failed_login_attempts integer DEFAULT 0,
    locked_until timestamp with time zone
);


ALTER TABLE public.users OWNER TO sutazai;

--
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
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sutazai
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: agent_executions id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions ALTER COLUMN id SET DEFAULT nextval('public.agent_executions_id_seq'::regclass);


--
-- Name: agent_health id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health ALTER COLUMN id SET DEFAULT nextval('public.agent_health_id_seq'::regclass);


--
-- Name: agents id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents ALTER COLUMN id SET DEFAULT nextval('public.agents_id_seq'::regclass);


--
-- Name: chat_history id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history ALTER COLUMN id SET DEFAULT nextval('public.chat_history_id_seq'::regclass);


--
-- Name: model_registry id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry ALTER COLUMN id SET DEFAULT nextval('public.model_registry_id_seq'::regclass);


--
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- Name: system_alerts id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts ALTER COLUMN id SET DEFAULT nextval('public.system_alerts_id_seq'::regclass);


--
-- Name: system_metrics id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_metrics ALTER COLUMN id SET DEFAULT nextval('public.system_metrics_id_seq'::regclass);


--
-- Name: tasks id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks ALTER COLUMN id SET DEFAULT nextval('public.tasks_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: agent_executions; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agent_executions (id, agent_id, task_id, status, input_data, output_data, execution_time, error_message, created_at) FROM stdin;
\.


--
-- Data for Name: agent_health; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agent_health (id, agent_id, status, last_heartbeat, cpu_usage, memory_usage, disk_usage, response_time, error_count, success_count, metadata, created_at, updated_at) FROM stdin;
1	1	healthy	2025-08-09 23:24:59.685658	0.00	0.00	0.00	0.000	0	0	{}	2025-08-09 23:24:59.685658	2025-08-09 23:24:59.685658
2	2	healthy	2025-08-09 23:24:59.685658	0.00	0.00	0.00	0.000	0	0	{}	2025-08-09 23:24:59.685658	2025-08-09 23:24:59.685658
3	3	healthy	2025-08-09 23:24:59.685658	0.00	0.00	0.00	0.000	0	0	{}	2025-08-09 23:24:59.685658	2025-08-09 23:24:59.685658
4	4	healthy	2025-08-09 23:24:59.685658	0.00	0.00	0.00	0.000	0	0	{}	2025-08-09 23:24:59.685658	2025-08-09 23:24:59.685658
5	5	healthy	2025-08-09 23:24:59.685658	0.00	0.00	0.00	0.000	0	0	{}	2025-08-09 23:24:59.685658	2025-08-09 23:24:59.685658
6	4	healthy	2025-08-09 23:25:19.417967	25.50	512.70	0.00	0.125	0	0	{}	2025-08-09 23:25:19.417967	2025-08-09 23:25:19.417967
\.


--
-- Data for Name: agents; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.agents (id, name, type, description, endpoint, port, is_active, capabilities, created_at) FROM stdin;
1	health-monitor	monitoring	\N	http://health-monitor:8080	10210	t	["health_check", "metrics"]	2025-08-09 11:48:02.541621
2	task-coordinator	orchestration	\N	http://task-coordinator:8080	10450	t	["task_routing", "scheduling"]	2025-08-09 11:48:02.541621
3	ollama-service	llm	\N	http://ollama:11434	11434	t	["text_generation", "chat"]	2025-08-09 11:48:02.541621
4	hardware-resource-optimizer	optimization	\N	http://hardware-resource-optimizer:8080	11110	t	["resource_monitoring", "optimization"]	2025-08-09 11:48:02.541621
5	ai-agent-orchestrator	orchestration	\N	http://ai-agent-orchestrator:8080	8589	t	["agent_coordination", "task_routing"]	2025-08-09 11:48:02.541621
\.


--
-- Data for Name: chat_history; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.chat_history (id, user_id, message, response, agent_used, tokens_used, response_time, created_at) FROM stdin;
\.


--
-- Data for Name: model_registry; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.model_registry (id, model_name, model_type, size_mb, status, ollama_status, usage_count, last_used, file_path, parameters, capabilities, created_at, updated_at) FROM stdin;
1	tinyllama	llm	637.00	active	loaded	0	\N	\N	{"parameters": "1.1B", "quantization": "Q4_0", "context_length": 2048}	[]	2025-08-09 23:24:07.022462	2025-08-09 23:24:07.022462
\.


--
-- Data for Name: sessions; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.sessions (id, user_id, token, expires_at, is_active, user_agent, ip_address, created_at, last_accessed) FROM stdin;
\.


--
-- Data for Name: system_alerts; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.system_alerts (id, alert_type, severity, title, description, source, status, resolved_at, resolved_by, metadata, created_at) FROM stdin;
1	database_test	info	Database Schema Initialization Complete	All tables created and initialized successfully	dba_admin	active	\N	\N	{}	2025-08-09 23:25:19.417967
\.


--
-- Data for Name: system_metrics; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.system_metrics (id, metric_name, metric_value, tags, recorded_at) FROM stdin;
\.


--
-- Data for Name: tasks; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.tasks (id, title, description, agent_id, user_id, status, priority, payload, result, error_message, created_at, started_at, completed_at) FROM stdin;
1	QA Test Task	Testing integer ID relationships	\N	1	pending	5	{}	\N	\N	2025-08-10 19:26:58.891629	\N	\N
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: sutazai
--

COPY public.users (id, username, email, password_hash, is_active, created_at, updated_at, is_admin, last_login, failed_login_attempts, locked_until) FROM stdin;
2	system	system@sutazai.local	$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewjyJyM7QK8kL5yC	t	2025-08-09 08:33:14.678865+00	2025-08-09 08:33:14.678865+00	f	\N	0	\N
4	testuser	test@example.com	$2b$12$Bb3F5df4b.a8srdVrs8tDeORK/f0.tQbfAMhf.WKHEdQAiGHHDtOO	t	2025-08-09 11:35:06.146473+00	2025-08-09 11:35:28.8902+00	f	2025-08-09 11:35:28.890943+00	0	\N
5	testuser3	test3@example.com	$2b$12$evgZb/VZLlUNKtPlMoBToO6OMYuRt63k9k0LwhzbOsModyr41UONW	t	2025-08-09 11:46:54.876466+00	2025-08-09 11:47:08.542303+00	f	2025-08-09 11:47:08.543288+00	0	\N
1	admin	admin@sutazai.local	$2b$12$NPu5JmL4gBSR21BQRYDoW.xzKoWyatlrbchYKuQes7kpsLb2zoBNG	t	2025-08-09 08:33:14.678865+00	2025-08-09 23:24:59.685658+00	t	\N	2	\N
7	dba_test	dba_test@sutazai.local	test_hash	t	2025-08-09 23:25:19.417967+00	2025-08-09 23:25:19.417967+00	f	\N	0	\N
\.


--
-- Name: agent_executions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agent_executions_id_seq', 1, false);


--
-- Name: agent_health_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agent_health_id_seq', 6, true);


--
-- Name: agents_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.agents_id_seq', 5, true);


--
-- Name: chat_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.chat_history_id_seq', 1, false);


--
-- Name: model_registry_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.model_registry_id_seq', 1, true);


--
-- Name: sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.sessions_id_seq', 1, false);


--
-- Name: system_alerts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.system_alerts_id_seq', 1, true);


--
-- Name: system_metrics_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.system_metrics_id_seq', 1, false);


--
-- Name: tasks_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.tasks_id_seq', 1, true);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: sutazai
--

SELECT pg_catalog.setval('public.users_id_seq', 8, true);


--
-- Name: agent_executions agent_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);


--
-- Name: agent_health agent_health_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health
    ADD CONSTRAINT agent_health_pkey PRIMARY KEY (id);


--
-- Name: agents agents_name_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_name_key UNIQUE (name);


--
-- Name: agents agents_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_pkey PRIMARY KEY (id);


--
-- Name: chat_history chat_history_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history
    ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);


--
-- Name: model_registry model_registry_model_name_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry
    ADD CONSTRAINT model_registry_model_name_key UNIQUE (model_name);


--
-- Name: model_registry model_registry_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.model_registry
    ADD CONSTRAINT model_registry_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_token_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_token_key UNIQUE (token);


--
-- Name: system_alerts system_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts
    ADD CONSTRAINT system_alerts_pkey PRIMARY KEY (id);


--
-- Name: system_metrics system_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_metrics
    ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);


--
-- Name: tasks tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: idx_agent_executions_agent_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_executions_agent_id ON public.agent_executions USING btree (agent_id);


--
-- Name: idx_agent_health_agent_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_agent_id ON public.agent_health USING btree (agent_id);


--
-- Name: idx_agent_health_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agent_health_status ON public.agent_health USING btree (status);


--
-- Name: idx_agents_type; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_agents_type ON public.agents USING btree (type);


--
-- Name: idx_chat_history_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_chat_history_user_id ON public.chat_history USING btree (user_id);


--
-- Name: idx_model_registry_name; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_model_registry_name ON public.model_registry USING btree (model_name);


--
-- Name: idx_sessions_token; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_token ON public.sessions USING btree (token);


--
-- Name: idx_sessions_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_sessions_user_id ON public.sessions USING btree (user_id);


--
-- Name: idx_system_alerts_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_alerts_status ON public.system_alerts USING btree (status);


--
-- Name: idx_system_metrics_name; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_metrics_name ON public.system_metrics USING btree (metric_name);


--
-- Name: idx_system_metrics_recorded_at; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_system_metrics_recorded_at ON public.system_metrics USING btree (recorded_at);


--
-- Name: idx_tasks_agent_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_agent_id ON public.tasks USING btree (agent_id);


--
-- Name: idx_tasks_status; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_status ON public.tasks USING btree (status);


--
-- Name: idx_tasks_user_id; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_tasks_user_id ON public.tasks USING btree (user_id);


--
-- Name: idx_users_email; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_users_email ON public.users USING btree (email);


--
-- Name: idx_users_is_admin; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_users_is_admin ON public.users USING btree (is_admin) WHERE (is_admin = true);


--
-- Name: idx_users_username; Type: INDEX; Schema: public; Owner: sutazai
--

CREATE INDEX idx_users_username ON public.users USING btree (username);


--
-- Name: agents update_agents_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON public.agents FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: tasks update_tasks_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON public.tasks FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: users update_users_updated_at; Type: TRIGGER; Schema: public; Owner: sutazai
--

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: agent_executions agent_executions_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id);


--
-- Name: agent_executions agent_executions_task_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_executions
    ADD CONSTRAINT agent_executions_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.tasks(id);


--
-- Name: agent_health agent_health_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.agent_health
    ADD CONSTRAINT agent_health_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id) ON DELETE CASCADE;


--
-- Name: chat_history chat_history_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.chat_history
    ADD CONSTRAINT chat_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: system_alerts system_alerts_resolved_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.system_alerts
    ADD CONSTRAINT system_alerts_resolved_by_fkey FOREIGN KEY (resolved_by) REFERENCES public.users(id);


--
-- Name: tasks tasks_agent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES public.agents(id);


--
-- Name: tasks tasks_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sutazai
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT tasks_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

--
-- PostgreSQL database cluster dump complete
--

