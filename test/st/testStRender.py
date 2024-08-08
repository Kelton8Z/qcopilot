import streamlit as st
import streamlit.components.v1 as components

# sources = [{"filename": "content"}]

# st.markdown(sources)

st.set_page_config(layout='wide')

st.markdown('<center><h2>Sample iframes</h2></center>', unsafe_allow_html=True)

cols = st.columns([1, 1])

# left
link1 = "https://chitu-ai.feishu.cn/wiki/Fdomws4MBiWOu7kUbOqc9B2ynBh#CZ3MdZVm7oO32vxPQdkcbK3InQh"
with cols[0]:
    components.iframe(link1, height=400, width=500)

# right
link2 = "https://chitu-ai.feishu.cn/wiki/SphbwVnJ6iSGCKk80V2cMhBHnQd"
with cols[1]:
    components.iframe(link2, height=400, width=500)