# --- Streamlit App: CTDA AI Assistant ---
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Set wide layout and title
st.set_page_config(page_title="📄 CTDA AI Assistant", layout="wide")
st.title("📄 CTDA AI Assistant")

BASE_URL = "http://127.0.0.1:8000/"

# --- Auth ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""

if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["admin", "user"])

    # Simple hardcoded check
    if st.button("Login"):
        if username == password and username == role:  # Optional: use user-role map instead
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = role
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

username = st.session_state.username
role = st.session_state.role

# --- Tabs ---
tabs = ["Upload Document", "Conversational QA"]
if role == "admin":
    tabs.extend(["Session History", "Health Check"])
selected_tab = st.tabs(tabs)

# --- Tab: Upload Document ---
with selected_tab[0]:
    st.header("📤 Upload Document")
    docname = st.text_input("Document Name")
    instruction = st.text_area("Instructions")
    include_summary = st.checkbox("Include AI Summary")
    summary_length = st.slider("Summary Length", 1, 10, 4)
    file = st.file_uploader("Upload File")

    if st.button("Upload") and file:
        files = {"file": (file.name, file.getvalue())}
        data = {
            "username": username,
            "docname": docname,
            "instruction": instruction,
            "include_summary": str(include_summary),
            "summary_length": str(summary_length)
        }
        res = requests.post(f"{BASE_URL}/upload_document/", data=data, files=files)
        if res.ok:
            st.success("✅ Document uploaded.")
            if include_summary:
                result = res.json()
                st.markdown("### Summary")
                st.write(result.get("summary", "No summary generated"))
        else:
            st.error(res.json().get("detail", "Upload failed"))

# --- Tab: Conversational QA ---
with selected_tab[1]:
    st.title("💬 Conversational Q&A")

    # Sidebar style area
    with st.sidebar:
        st.title("📂 Document Manager")
        st.markdown("---")
        docs = []
        try:
            res = requests.get(f"{BASE_URL}/list_documents/")
            if res.ok:
                docs = [doc['docname'] for doc in res.json()]
        except:
            st.error("Unable to fetch documents")

        docname = st.selectbox("📄 Select Document", docs)

        if st.button("🆕 New Chat"):
            st.session_state.chat = []
            st.session_state.session_id = f"{username}_{datetime.utcnow().isoformat()}"
            st.session_state.chat.append({"role": "assistant", "content": "👋 How may I help you today?"})
            st.rerun()

        if st.button("🗑️ Delete Selected Document") and docname:
            res = requests.delete(f"{BASE_URL}/delete_document/{username}/{docname}")
            if res.ok:
                st.success("Deleted")
                st.rerun()
            else:
                st.error("Delete failed")

        if role == "admin" and st.button("❌ Delete All Documents"):
            res = requests.delete(f"{BASE_URL}/delete_all_documents/")
            if res.ok:
                st.success("All documents deleted")
                st.rerun()
            else:
                st.error("Failed to delete all")

    if docname:
        if "chat" not in st.session_state:
            st.session_state.chat = []
            st.session_state.chat.append({"role": "assistant", "content": "👋 How may I help you today?"})
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"{username}_{datetime.utcnow().isoformat()}"

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask a question...")

        if query:
            st.session_state.chat.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            payload = {
                "username": username,
                "session_id": st.session_state.session_id,
                "query": query,
                "docnames": [docname]
            }

            res = requests.post(f"{BASE_URL}/conversational_qa/", data=payload)
            if res.ok:
                answer = res.json().get("answer", "No response")
                st.session_state.chat.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("💬 Give Feedback"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍", key=f"like_{len(st.session_state.chat)}"):
                                res = requests.post(f"{BASE_URL}/submit_feedback/", data={
                                    "username": username,
                                    "prompt": query,
                                    "response": answer,
                                    "feedback": "like"
                                })
                                if res.ok:
                                    st.success("Feedback recorded")
                                else:
                                    st.error(f"Failed to record feedback: {res.json().get('detail', 'Unknown error')}")
                        with col2:
                            if st.button("👎", key=f"dislike_{len(st.session_state.chat)}"):
                                with st.form(key=f"feedback_form_{len(st.session_state.chat)}"):
                                    comment = st.text_input("Comment", key=f"comment_{len(st.session_state.chat)}")
                                    submit_button = st.form_submit_button("Submit Feedback")
                                    if submit_button and comment:
                                        res = requests.post(f"{BASE_URL}/submit_feedback/", data={
                                            "username": username,
                                            "prompt": query,
                                            "response": answer,
                                            "feedback": f"dislike: {comment}"
                                        })
                                        if res.ok:
                                            st.success("Feedback recorded")
                                        else:
                                            st.error(f"Failed to record feedback: {res.json().get('detail', 'Unknown error')}")
            else:
                st.error("Failed to get response from bot")
    else:
        st.info("Please select a document from the sidebar to start chatting.")

# --- Tab: Session History (Admin Only) ---
if role == "admin":
    with selected_tab[2]:
        st.header("📜 Session History")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_user = st.text_input("Filter by Username", value=username)
        with filter_col2:
            filter_date = st.date_input("Filter by Date")

        params = {"username": filter_user}
        if filter_date:
            params["date"] = str(filter_date)

        try:
            res = requests.get(f"{BASE_URL}/session_history/", params=params)
            if res.ok:
                sessions = res.json()

                if sessions:
                    grouped = {}
                    for s in sessions:
                        grouped.setdefault(s["session_id"], []).append(s)

                    session_ids = list(grouped.keys())
                    selected_session = st.selectbox("Select Session", session_ids)

                    if selected_session:
                        records = grouped[selected_session]
                        for r in records:
                            st.markdown(f"**🙋 User:** {r['query']}")
                            st.markdown(f"**🤖 Bot:** {r['response']}")
                            st.caption(f"🕒 {r['timestamp']} | 📄 {r['document_name']}")
                            st.markdown("---")

                        if st.button("❌ Delete This Session"):
                            res = requests.delete(f"{BASE_URL}/delete_session/{selected_session}")
                            if res.ok:
                                st.success("Session deleted")
                                st.rerun()
                            else:
                                st.error("Delete failed")

                    if st.button("📥 Download All Sessions"):
                        df = pd.DataFrame(sessions)
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV", csv, file_name="session_history.csv", mime="text/csv")

                    if st.button("🚨 Delete All Sessions"):
                        res = requests.delete(f"{BASE_URL}/delete_all_sessions/", params={"username": filter_user})
                        if res.ok:
                            st.success("All sessions deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete all sessions")
                else:
                    st.info("No session history found.")
        except Exception as e:
            st.error(f"Failed to fetch sessions: {str(e)}")

# --- Tab: Health Check ---
if role == "admin":
    with selected_tab[3]:
        st.header("🩺 System Health Dashboard")

        # Refresh Button
        col1, col2 = st.columns([1, 8])
        with col1:
            if st.button("🔄"):
                st.rerun()
        with col2:
            st.markdown("### Click to refresh system health")

        # Fetch health status
        res = requests.get(f"{BASE_URL}/health/")
        if res.ok:
            health = res.json()
            status = health.get("overall_status", "unknown").lower()

            # Overall Status Display
            if status == "healthy":
                st.success("✅ **System is Healthy**")
            elif status == "degraded":
                st.warning("⚠️ **System is Degraded**")
            else:
                st.error(f"❌ **System Status: {status.upper()}**")

            st.markdown("---")
            st.subheader("📊 Core Service Status")

            # Display summary of core services only (e.g., LLMService, DBService, etc.)
            components = health.get("components", {})
            for service, metrics in components.items():
                if not isinstance(metrics, dict):
                    continue

                # Determine health status
                service_status = metrics.get("status", "unknown").lower()
                bar_val = 100 if service_status == "healthy" else 50 if service_status == "degraded" else 0

                with st.container():
                    st.markdown(f"**🔹 {service.replace('_', ' ').title()}**")
                    if service_status == "healthy":
                        st.success("🟢 Healthy")
                    elif service_status == "degraded":
                        st.warning("🟠 Degraded")
                    else:
                        st.error("🔴 Unhealthy")

                    st.progress(bar_val / 100)
                    st.markdown(f"**{bar_val}%**")

            st.markdown("---")
            st.subheader("📄 Additional Information")

            # Additional non-component fields
            extra_info = {
                k: v for k, v in health.items()
                if k not in ["overall_status", "components"]
            }
            for key, value in extra_info.items():
                with st.container():
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    if isinstance(value, (int, float)):
                        pct = min(max(float(value), 0.0), 100.0)
                        st.progress(pct / 100)
                        st.info(f"{pct} %")
                    else:
                        st.info(str(value))

            with st.expander("🔍 View Raw JSON"):
                st.json(health)
        else:
            st.error("🚫 Failed to fetch health check status")