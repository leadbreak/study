import streamlit as st

def scroll() -> None:
    
    js = '''
    <script>
        var body = window.parent.document.getElementById(".main");
        body.scrollTop += 9999999999;
    </script>
    '''
    
    st.components.v1.html(js)

st.title("Streamlit Scroll Test")

with st.container():
    st.header("첫 번째 컨테이너")
    
    for i in range(20):
        st.write(f"1st Area : {i}")

with st.container():
    st.header("두 번째 컨테이너")
    for j in range(20):
        st.write(f"2nd Area : {j}")
    
    
st.sidebar.button('scroll to bottom of main', on_click=scroll)
