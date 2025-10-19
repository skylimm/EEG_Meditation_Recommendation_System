import os
import streamlit as st

# def get_user_folder(base_dir="C:/Users/slex8/OneDrive - Nanyang Technological University/UNI/FYP/data"):
#
#     user_name = st.text_input("Enter your name (this helps us label your EEG file): ", key="username")
#     folder_path = None
#
#     if user_name:
#         folder_path = os.path.join(base_dir, user_name)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         # st.caption(f"Data can be found in: `{folder_path}`")
#         st.write("Welcome " + user_name + "!")
#     return folder_path
import os
import streamlit as st

def get_user_folder(base_dir="./data"):
    # Placeholder to show/hide the input row
    name_ui = st.empty()

    # Callback: move input -> username
    def save_name():
        name = (st.session_state.get("username_input") or "").strip()
        if name:
            st.session_state["username"] = name

    # Show input only when username not set
    if "username" not in st.session_state:
        with name_ui.container():
            st.text_input(
                "Enter your name (this helps us label your EEG file):",
                key="username_input",
                on_change=save_name,
            )

    # If username is now set (after Enter), hide the input
    if "username" in st.session_state:
        # remove the input UI
        name_ui.empty()

        user_name = st.session_state["username"]
        folder_path = os.path.join(base_dir, user_name)
        os.makedirs(folder_path, exist_ok=True)

        # Styled welcome (custom font look via CSS)
        st.markdown("""
        <style>
            .welcome-title { font-size: 2rem; margin: 0 0 4px 0; }
            .welcome-sub { font-size: 0.95rem; color: #666; margin: 0; }
            .highlight-name {
                background: linear-gradient(90deg, #4CAF50, #2E7D32);
                -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 700;
            }
            .highlight-welcome {
                background: linear-gradient(90deg, #3D85C6, #00498D);
                -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 700;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
                <div class="welcome-title"><span class="highlight-welcome">Welcome </span><span class="highlight-name">{user_name},</span></div>
            """,
            unsafe_allow_html=True
        )


        return folder_path

    # No folder until name entered
    return None
