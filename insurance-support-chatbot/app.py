# Insurance Customer Care Assistant — Streamlit Chatbot
# Databricks Custom App with Conversation History
 
import os
import uuid
import streamlit as st
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
 
# WorkspaceClient handles OAuth automatically in Databricks Apps context
_w            = WorkspaceClient()
ENDPOINT_NAME = os.environ.get(
    "AGENT_ENDPOINT_NAME",
    "agents_main-insurance_support-insurance_support_agent",
)
 
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
 
 
# ── Session state bootstrap ────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "conversations":     {},
        "active_session_id": str(uuid.uuid4()),
        "messages":          [],
        "turn_count":        0,
        "pending_query":     None,
        "endpoint_ok":       None,
        "load_session_id":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
_init_session()
 
 
# ── Conversation helpers ───────────────────────────────────────────────────────
def _first_user_message(messages):
    for m in messages:
        if m.get("role") == "user":
            txt = m["content"].strip()
            return (txt[:45] + "...") if len(txt) > 45 else txt
    return "Untitled"
 
 
def save_current_conversation():
    sid  = st.session_state["active_session_id"]
    msgs = st.session_state["messages"]
    if not msgs:
        return
    title    = _first_user_message(msgs)
    existing = st.session_state["conversations"].get(sid, {})
    st.session_state["conversations"][sid] = {
        "id":         sid,
        "title":      title,
        "messages":   list(msgs),
        "created_at": existing.get("created_at", datetime.now().strftime("%b %d, %H:%M")),
        "turn_count": st.session_state["turn_count"],
    }
 
 
def new_session():
    save_current_conversation()
    st.session_state["active_session_id"] = str(uuid.uuid4())
    st.session_state["messages"]          = []
    st.session_state["turn_count"]        = 0
    st.session_state["pending_query"]     = None
 
 
def load_conversation(sid):
    save_current_conversation()
    conv = st.session_state["conversations"].get(sid)
    if not conv:
        return
    st.session_state["active_session_id"] = sid
    st.session_state["messages"]          = list(conv["messages"])
    st.session_state["turn_count"]        = conv.get("turn_count", 0)
    st.session_state["pending_query"]     = None
 
 
def delete_conversation(sid):
    st.session_state["conversations"].pop(sid, None)
    if st.session_state["active_session_id"] == sid:
        new_session()
 
 
def delete_all_conversations():
    st.session_state["conversations"]     = {}
    st.session_state["active_session_id"] = str(uuid.uuid4())
    st.session_state["messages"]          = []
    st.session_state["turn_count"]        = 0
    st.session_state["pending_query"]     = None
 
 
if st.session_state["load_session_id"]:
    load_conversation(st.session_state["load_session_id"])
    st.session_state["load_session_id"] = None
 
 
# ── Endpoint helpers ───────────────────────────────────────────────────────────
_ROLE_MAP = {
    "user":      ChatMessageRole.USER,
    "assistant": ChatMessageRole.ASSISTANT,
    "system":    ChatMessageRole.SYSTEM,
}
 
 
def call_agent(messages):
    """
    Call the serving endpoint via SDK — handles OAuth automatically.
    Strips the internal 'ts' timestamp field before sending so only
    role and content are forwarded to predict().
    """
    try:
        sdk_msgs = [
            ChatMessage(
                role    = _ROLE_MAP.get(m["role"], ChatMessageRole.USER),
                content = m["content"],
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
    """
    FIX: Actually verify the endpoint is READY via SDK instead of just
    checking if the name string is non-empty (which is always True).
    """
    try:
        ep    = _w.serving_endpoints.get(name=ENDPOINT_NAME)
        state = str(ep.state.ready) if ep.state else ""
        return "READY" in state.upper()
    except Exception:
        return False
 
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Insurance Assistant")
    st.caption("Powered by Databricks Multi-Agent System")
    st.divider()
 
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
 
    st.markdown("**Current Session**")
    st.code(st.session_state["active_session_id"][:18] + "...", language=None)
    turn_count = st.session_state["turn_count"]
    st.caption(f"Turns: {turn_count}")
 
    if st.button("➕ New Conversation", use_container_width=True):
        new_session()
        st.rerun()
 
    st.divider()
 
    st.markdown("**📂 Conversation History**")
 
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
            title   = conv.get("title", "Untitled")
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
        if st.session_state["messages"] else "New Conversation"
    )
    st.caption(f"**{active_title}** · Multi-turn conversations supported")
with col_clear:
    st.write("")
    if st.button("🗑️ Clear", help="Clear current chat"):
        st.session_state["messages"]   = []
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
 
user_query = (
    st.session_state.pop("pending_query", None)
    or st.chat_input("Type your question here...", key="chat_input")
)
 
if user_query:
    ts_now = datetime.now().strftime("%H:%M")
    st.session_state["messages"].append({"role": "user", "content": user_query, "ts": ts_now})
    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(ts_now)
 
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your request..."):
            agent_response = call_agent(st.session_state["messages"])
        st.markdown(agent_response)
        ts_resp = datetime.now().strftime("%H:%M")
        st.caption(ts_resp)
 
    st.session_state["messages"].append({
        "role": "assistant", "content": agent_response, "ts": ts_resp
    })
    st.session_state["turn_count"] += 1
    save_current_conversation()
    st.rerun()
