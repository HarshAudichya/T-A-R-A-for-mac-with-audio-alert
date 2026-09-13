import os
import streamlit as st
from PIL import Image
from gui.gui_manager import GUIManager

def main():
    logo_path = os.path.join(os.path.dirname(__file__), "gui", "assets", "tara_favicon.png")
    icon = Image.open(logo_path) if os.path.exists(logo_path) else "👩‍🚀"
    st.set_page_config(
        page_title="TARA — Task-aware Astronaut Recognition Assistant",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    gui = GUIManager()
    gui.render()

if __name__ == "__main__":
    main()
