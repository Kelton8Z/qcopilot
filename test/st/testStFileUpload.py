import streamlit as st

languages = {
    "EN": {
        "button": "Browse Files",
        "instructions": "Drag and drop files here",
        "limits": "Limit 200MB per file",
    },
    "CN": {
        "button": "本地上传",
        "instructions": "拖放",
        "limits": "每个文件最大200MB",
    },
}
lang = "CN"


hide_label = (
    """
<style>
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
       color:white;
    }
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
        content: "BUTTON_TEXT";
        color:black;
        display: block;
        position: absolute;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span::after {
       content:"INSTRUCTIONS_TEXT";
       visibility:visible;
       display:block;
    }
     div[data-testid="stFileDropzoneInstructions"]>div>small {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>small::before {
       content:"FILE_LIMITS";
       visibility:visible;
       display:block;
    }
</style>
""".replace(
        "BUTTON_TEXT", languages.get(lang).get("button")
    )
    .replace("INSTRUCTIONS_TEXT", languages.get(lang).get("instructions"))
    .replace("FILE_LIMITS", languages.get(lang).get("limits"))
)

if "cnt" not in st.session_state:
    st.session_state.cnt = 0
else:
    st.session_state.cnt += 1

st.markdown(hide_label, unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader(label="上传临时文件", accept_multiple_files=True) 
if st.session_state.cnt > 0:
    assert(st.session_state.uploaded_files == uploaded_files, f'{len(st.session_state.uploaded_files)} vs {len(uploaded_files)}')
st.session_state.uploaded_files = uploaded_files
st.write(st.session_state.uploaded_files)
if st.button(label="rerun"):
    st.rerun()