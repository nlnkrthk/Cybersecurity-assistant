"""
Sign in / create account. Shown instead of the app until the user is signed in.
"""
import streamlit as st

from api_client import ApiError
from components import get_client, start_session

ss = st.session_state


def submit(action: str, username: str, password: str, confirm: str = None) -> None:
    if not username.strip() or not password:
        st.error("Enter a username and a password.", icon=":material/error:")
        return
    if confirm is not None and password != confirm:
        st.error("The passwords do not match.", icon=":material/error:")
        return
    try:
        client = get_client()
        session = client.signup(username.strip(), password) if action == "signup" else \
            client.login(username.strip(), password)
    except ApiError as exc:
        message = exc.detail if exc.status in (401, 409, 422, 429) else exc.message
        st.error(message, icon=":material/error:")
        return
    start_session(session)
    st.rerun()


_, middle, _ = st.columns([1, 1.4, 1])
with middle:
    st.header(":material/shield: Incident Response Assistant")
    st.markdown('<div class="page-subtitle">Sign in to investigate incidents with the AI assistant. '
                "Your conversations are saved to your account.</div>", unsafe_allow_html=True)
    notice = ss.pop("auth_notice", None)
    if notice:
        st.info(notice, icon=":material/info:")

    sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"], key="auth_tab")
    with sign_in_tab:
        with st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            if st.form_submit_button("Sign in", type="primary", icon=":material/login:", width="stretch"):
                submit("login", username, password)
    with sign_up_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", help="3-32 characters: letters, digits, '.', '_' or '-'.",
                                         autocomplete="username")
            new_password = st.text_input("Choose a password", type="password", help="At least 8 characters.",
                                         autocomplete="new-password")
            confirm = st.text_input("Repeat the password", type="password", autocomplete="new-password")
            if st.form_submit_button("Create account", type="primary", icon=":material/person_add:", width="stretch"):
                submit("signup", new_username, new_password, confirm)
    st.caption("Accounts are stored by this application's own server; passwords are kept only as salted "
               "hashes. Closing the tab or refreshing the page signs you out.")
