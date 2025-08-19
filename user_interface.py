import streamlit as st

def main():
    st.title('AI Doctor Assitant')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        st.chat_message(message['role'], message['content'])
    
    prompt = st.chat_input('Ask Anything...')
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        response = "Hi! I am your AI Doctor"
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == '__main__':
    main()