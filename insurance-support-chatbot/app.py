# Insurance Customer Care Assistant — Streamlit Chatbot
# Databricks Custom App with Lakebase-persisted Conversation History

import os
import json
import uuid
import psycopg2
import streamlit as st
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# ── Databricks clients ─────────────────────────────────────────────────────────
_w = WorkspaceClient()
APP_SP_NAME = _w.current_user.me().user_name  # app SP UUID — PostgreSQL role name


def _get_current_user() -> str:
    """Logged-in user's email from Databricks Apps request headers."""
    try:
        headers = st.context.headers
        email = (
            headers.get("X-Forwarded-Email")
            or headers.get("X-Databricks-User-Email")
            or headers.get("X-Forwarded-User")
            or ""
        )
        if email:
            return email
    except Exception:
        pass
    return APP_SP_NAME


CURRENT_USER = _get_current_user()  # email for data isolation

ENDPOINT_NAME = os.environ.get(
    "AGENT_ENDPOINT_NAME",
    "agents_main-insurance_support-insurance_support_agent",
)

# ── Lakebase connection config (set via app.yaml env vars) ─────────────────────
LAKEBASE_ENDPOINT = os.environ.get(
    "LAKEBASE_ENDPOINT",
    "projects/insurance-support-pg/branches/production/endpoints/primary",
)
LAKEBASE_HOST = os.environ.get(
    "LAKEBASE_HOST",
    "ep-calm-cell-d1wr2dke.database.us-west-2.cloud.databricks.com",
)
LAKEBASE_DB = "databricks_postgres"

st.set_page_config(
    page_title="Insurance Customer Care",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 6px;
        padding: 6px;
    }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) { background-color: #f0f4ff; }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarAssistant"]
    ) { background-color: #f9f9fb; }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) * { color: #1a1a1a !important; }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarAssistant"]
    ) * { color: #1a1a1a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Lakebase helpers ───────────────────────────────────────────────────────────


@st.cache_data(ttl=3000, show_spinner=False)
def _get_lakebase_token(_key: str = "app_sp") -> str:
    """App SP token — cached 50 min, keyed globally (not per user)."""
    cred = _w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT)
    return cred.token


def _get_conn() -> psycopg2.extensions.connection:
    """Connect as app SP — token and user must match."""
    return psycopg2.connect(
        host=LAKEBASE_HOST,
        port=5432,
        dbname=LAKEBASE_DB,
        user=APP_SP_NAME,  # ← app SP UUID (matches the token)
        password=_get_lakebase_token(),
        sslmode="require",
    )


def _ensure_table():
    """
    Create conversation_metadata table if it doesn't exist.
    Called once per session via st.cache_resource.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_metadata (
                    session_id   TEXT PRIMARY KEY,
                    user_email   TEXT NOT NULL,
                    title        TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ DEFAULT NOW(),
                    turn_count   INTEGER DEFAULT 0,
                    messages     TEXT
                )
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversation_metadata (user_email)
            """
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


@st.cache_resource
def _init_db_once():
    """Run table creation exactly once per app deployment."""
    try:
        _ensure_table()
        return True
    except Exception as e:
        return False


_db_ready = _init_db_once()


def db_load_conversations(user_email: str) -> dict:
    """Load all conversations for this user from Lakebase."""
    if not _db_ready:
        return {}
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, title, created_at, turn_count, messages
                    FROM   conversation_metadata
                    WHERE  user_email = %s
                    ORDER  BY updated_at DESC
                """,
                    (user_email,),
                )
                rows = cur.fetchall()

            convs = {}
            for sid, title, created_at, turn_count, messages_json in rows:
                convs[sid] = {
                    "id": sid,
                    "title": title or "Untitled",
                    "created_at": (
                        created_at.strftime("%b %d, %H:%M") if created_at else ""
                    ),
                    "turn_count": turn_count or 0,
                    "messages": json.loads(messages_json) if messages_json else [],
                }
            return convs
        finally:
            conn.close()
    except Exception as e:
        st.warning(f"Could not load conversation history: {e}")
        return {}


def db_save_conversation(conv: dict, user_email: str):
    """Upsert a conversation to Lakebase."""
    if not _db_ready or not conv.get("messages"):
        return
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_metadata
                        (session_id, user_email, title, turn_count, messages, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        title      = EXCLUDED.title,
                        turn_count = EXCLUDED.turn_count,
                        messages   = EXCLUDED.messages,
                        updated_at = NOW()
                """,
                    (
                        conv["id"],
                        user_email,
                        conv.get("title", "Untitled"),
                        conv.get("turn_count", 0),
                        json.dumps(conv.get("messages", [])),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        st.warning(f"Could not save conversation: {e}")


def db_delete_conversation(session_id: str):
    """
    Delete one conversation from Lakebase.
    Also clears LangGraph checkpoint data for this thread_id.
    """
    if not _db_ready:
        return
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_metadata WHERE session_id = %s",
                    (session_id,),
                )
                # Clear LangGraph checkpoint tables (thread_id = session_id)
                for tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                    cur.execute(
                        f"DELETE FROM public.{tbl} WHERE thread_id = %s",
                        (session_id,),
                    )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        st.warning(f"Could not delete from database: {e}")


def db_delete_all_conversations(user_email: str):
    """
    Delete all conversations for this user from Lakebase.
    Also clears their LangGraph checkpoint data.
    """
    if not _db_ready:
        return
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id FROM conversation_metadata WHERE user_email = %s",
                    (user_email,),
                )
                session_ids = [r[0] for r in cur.fetchall()]

                cur.execute(
                    "DELETE FROM conversation_metadata WHERE user_email = %s",
                    (user_email,),
                )
                for sid in session_ids:
                    for tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                        cur.execute(
                            f"DELETE FROM public.{tbl} WHERE thread_id = %s",
                            (sid,),
                        )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        st.warning(f"Could not delete all conversations: {e}")


# ── Session state bootstrap ────────────────────────────────────────────────────


def _init_session():
    defaults = {
        "conversations": {},
        "active_session_id": str(uuid.uuid4()),
        "messages": [],
        "turn_count": 0,
        "pending_query": None,
        "endpoint_ok": None,
        "load_session_id": None,
        "db_loaded": False,  # track whether DB conversations are loaded
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ── Load conversations from DB once per session ────────────────────────────────
if not st.session_state["db_loaded"]:
    persisted = db_load_conversations(CURRENT_USER)
    st.session_state["conversations"].update(persisted)
    st.session_state["db_loaded"] = True


# ── Conversation helpers ───────────────────────────────────────────────────────


def _first_user_message(messages):
    for m in messages:
        if m.get("role") == "user":
            txt = m["content"].strip()
            return (txt[:45] + "...") if len(txt) > 45 else txt
    return "Untitled"


def save_current_conversation():
    sid = st.session_state["active_session_id"]
    msgs = st.session_state["messages"]
    if not msgs:
        return
    title = _first_user_message(msgs)
    existing = st.session_state["conversations"].get(sid, {})
    conv = {
        "id": sid,
        "title": title,
        "messages": list(msgs),
        "created_at": existing.get(
            "created_at", datetime.now().strftime("%b %d, %H:%M")
        ),
        "turn_count": st.session_state["turn_count"],
    }
    st.session_state["conversations"][sid] = conv
    db_save_conversation(conv, CURRENT_USER)


def new_session():
    save_current_conversation()
    st.session_state["active_session_id"] = str(uuid.uuid4())
    st.session_state["messages"] = []
    st.session_state["turn_count"] = 0
    st.session_state["pending_query"] = None


def load_conversation(sid):
    save_current_conversation()
    conv = st.session_state["conversations"].get(sid)
    if not conv:
        return
    st.session_state["active_session_id"] = sid
    st.session_state["messages"] = list(conv["messages"])
    st.session_state["turn_count"] = conv.get("turn_count", 0)
    st.session_state["pending_query"] = None


def delete_conversation(sid):
    st.session_state["conversations"].pop(sid, None)
    db_delete_conversation(sid)
    if st.session_state["active_session_id"] == sid:
        new_session()


def delete_all_conversations():
    st.session_state["conversations"] = {}
    st.session_state["active_session_id"] = str(uuid.uuid4())
    st.session_state["messages"] = []
    st.session_state["turn_count"] = 0
    st.session_state["pending_query"] = None
    db_delete_all_conversations(CURRENT_USER)


if st.session_state["load_session_id"]:
    load_conversation(st.session_state["load_session_id"])
    st.session_state["load_session_id"] = None


# ── Endpoint helpers ───────────────────────────────────────────────────────────

_ROLE_MAP = {
    "user": ChatMessageRole.USER,
    "assistant": ChatMessageRole.ASSISTANT,
    "system": ChatMessageRole.SYSTEM,
}


def call_agent(messages):
    """
    Call the serving endpoint via SDK — handles OAuth automatically.
    Strips the internal 'ts' timestamp field before forwarding.
    """
    try:
        sdk_msgs = [
            ChatMessage(
                role=_ROLE_MAP.get(m["role"], ChatMessageRole.USER),
                content=m["content"],
            )
            for m in messages
        ]
        resp = _w.serving_endpoints.query(name=ENDPOINT_NAME, messages=sdk_msgs)
        if resp.choices:
            return resp.choices[0].message.content
        return f"Unexpected response format: {str(resp)[:300]}"
    except Exception as e:
        return f"❌ Error calling agent: {str(e)}"


def check_endpoint_health():
    try:
        ep = _w.serving_endpoints.get(name=ENDPOINT_NAME)
        state = str(ep.state.ready) if ep.state else ""
        return "READY" in state.upper()
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Insurance Assistant")
    st.caption("Powered by Databricks Multi-Agent System")
    st.divider()

    # ── Logged-in user ─────────────────────────────────────────────────────────
    st.markdown("**👤 Logged in as**")
    st.caption(CURRENT_USER)
    st.divider()

    # ── Endpoint health ────────────────────────────────────────────────────────
    if st.session_state["endpoint_ok"] is None:
        with st.spinner("Checking endpoint..."):
            st.session_state["endpoint_ok"] = check_endpoint_health()
    if st.session_state["endpoint_ok"]:
        st.markdown(":green[🟢 **Endpoint Ready**]")
    else:
        st.markdown(":red[🔴 **Endpoint Unreachable**]")
        if st.button("🔄 Retry", use_container_width=True):
            st.session_state["endpoint_ok"] = None
            st.rerun()
    st.caption(f"`{ENDPOINT_NAME}`")
    st.divider()

    # ── DB status ──────────────────────────────────────────────────────────────
    if _db_ready:
        st.markdown(":green[🗄️ **History: Lakebase connected**]")
    else:
        st.markdown(":orange[⚠️ **History: DB unavailable (session only)**]")
    st.divider()

    # ── Current session ────────────────────────────────────────────────────────
    st.markdown("**Current Session**")
    st.code(st.session_state["active_session_id"][:18] + "...", language=None)
    turn_count = st.session_state["turn_count"]
    st.caption(f"Turns: {turn_count}")

    if st.button("➕ New Conversation", use_container_width=True):
        new_session()
        st.rerun()

    st.divider()

    # ── Conversation history ───────────────────────────────────────────────────
    st.markdown("**📂 Conversation History**")
    st.caption(f"Showing history for: `{CURRENT_USER}`")

    history = {
        sid: conv
        for sid, conv in st.session_state["conversations"].items()
        if sid != st.session_state["active_session_id"] and conv.get("messages")
    }

    if not history:
        st.caption("No past conversations yet.")
    else:
        if st.button(
            "🗑️ Delete All Conversations",
            use_container_width=True,
            type="secondary",
        ):
            delete_all_conversations()
            st.rerun()

        st.markdown("---")

        sorted_convs = sorted(
            history.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        )

        for sid, conv in sorted_convs:
            title = conv.get("title", "Untitled")
            created = conv.get("created_at", "")
            n_turns = conv.get("turn_count", 0)

            col_title, col_kebab = st.columns([5, 1])

            with col_title:
                if st.button(
                    f"💬 {title}",
                    key=f"load_{sid}",
                    use_container_width=True,
                    help=f"{n_turns} turn(s) · {created}",
                ):
                    st.session_state["load_session_id"] = sid
                    st.rerun()

            with col_kebab:
                with st.popover("⋮", help="Options"):
                    st.markdown(f"**{title[:30]}**")
                    st.caption(f"{n_turns} turn(s) · {created}")
                    st.markdown("---")
                    if st.button(
                        "🗑️ Delete this conversation",
                        key=f"del_{sid}",
                        use_container_width=True,
                        type="primary",
                    ):
                        delete_conversation(sid)
                        st.rerun()

    st.divider()

    # ── Example queries ────────────────────────────────────────────────────────
    st.markdown("**💡 Try These Queries**")
    examples = [
        "What does life insurance cover?",
        "What is the status of my recent claim?",
        "What is my current premium amount?",
        "Show me my payment history.",
        "I want to speak to a human agent.",
        "What is a deductible in auto insurance?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["pending_query"] = ex

    st.divider()

    with st.expander("🤖 Agent Routing Info"):
        st.markdown(
            "**Supervisor** routes your query to:\n"
            "- 📋 **Policy Agent** — coverage & policy details\n"
            "- 💳 **Billing Agent** — payments & invoices\n"
            "- 🔖 **Claims Agent** — claim status & filing\n"
            "- ❓ **General Help** — FAQs via Vector Search\n"
            "- 👤 **Human Escalation** — live agent handoff"
        )

    st.divider()
    st.caption("© Insurance Customer Care · Databricks Apps")


# ── Main chat area ─────────────────────────────────────────────────────────────
col_header, col_clear = st.columns([5, 1])
with col_header:
    st.title("🛡️ Insurance Customer Care")
    active_title = (
        _first_user_message(st.session_state["messages"])
        if st.session_state["messages"]
        else "New Conversation"
    )
    st.caption(f"**{active_title}** · Multi-turn conversations supported")
with col_clear:
    st.write("")
    if st.button("🗑️ Clear", help="Clear current chat and delete from history"):
        sid = st.session_state["active_session_id"]
        db_delete_conversation(sid)
        st.session_state["conversations"].pop(sid, None)
        st.session_state["messages"] = []
        st.session_state["turn_count"] = 0
        st.rerun()

st.divider()

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Welcome to Insurance Customer Care!**\n\n"
            "I am your AI-powered assistant. I can help you with:\n"
            "- 📋 **Policy details** — coverage, deductibles, vehicle info\n"
            "- 💳 **Billing & payments** — premium amounts, payment history\n"
            "- 🔖 **Claims** — status, filing guidance, settlements\n"
            "- ❓ **General FAQs** — insurance concepts explained clearly\n\n"
            "How can I help you today?"
        )

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "ts" in msg:
            st.caption(msg["ts"])

user_query = st.session_state.pop("pending_query", None) or st.chat_input(
    "Type your question here...", key="chat_input"
)

if user_query:
    ts_now = datetime.now().strftime("%H:%M")
    st.session_state["messages"].append(
        {"role": "user", "content": user_query, "ts": ts_now}
    )
    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(ts_now)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your request..."):
            agent_response = call_agent(st.session_state["messages"])
        st.markdown(agent_response)
        ts_resp = datetime.now().strftime("%H:%M")
        st.caption(ts_resp)

    st.session_state["messages"].append(
        {"role": "assistant", "content": agent_response, "ts": ts_resp}
    )
    st.session_state["turn_count"] += 1
    save_current_conversation()  # saves to session state + Lakebase
    st.rerun()
