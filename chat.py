import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_message

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")
st.title("챗봇!!")
st.caption("소득세에 관련된 모든것을 답변해드립니다.")

if 'message_list' not in st.session_state:
  st.session_state.message_list = []
  
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
      st.write(message["content"])
      
if user_question := st.chat_input(placeholder="입력해주세요"):
  with st.chat_message("user"):
    st.write(user_question)
  st.session_state.message_list.append({"role": "user", "content": user_question})
  
  with st.spinner("답변 생성중..."):
    ai_message = get_ai_message(user_question)
    
    with st.chat_message("ai"):
      st.write(ai_message)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})
