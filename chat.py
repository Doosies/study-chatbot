import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA

load_dotenv()

def get_ai_message(user_message: str) -> str:
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")
  index_name = "inflearn1"
  data_base = PineconeVectorStore.from_existing_index(index_name,embedding)
  retriever = data_base.as_retriever(search_kwargs={'k': 10})

  llm = ChatOpenAI(model='gpt-4o')
  prompt = hub.pull("rlm/rag-prompt")

  dictionary = ["사람을 나타내는 표현 -> 거주자"]
  converted_prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 안아도 됩니다.
    사전: {dictionary}
    
    질문: {{question}}
  """)

  dictionary_chain = converted_prompt | llm | StrOutputParser()
  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
  )
  tax_chain = {"query": dictionary_chain} | qa_chain

  ai_response = tax_chain.invoke({"question": user_message})
  
  return ai_response['result']

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
