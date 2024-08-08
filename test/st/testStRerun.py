import streamlit as st

if "llm" not in st.session_state.keys(): 
    st.session_state.llm = "claude3.5"
        
llm = st.sidebar.selectbox(
        "模型切换",
        ("gpt4o", "Claude3.5"),
        index=1
    )
print(llm)

if llm!=st.session_state["llm"]:
    st.session_state["llm"] = llm
    st.rerun()
