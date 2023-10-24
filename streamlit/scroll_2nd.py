import streamlit as st

def scroll(js_path:str='main'
           ) -> None:
    
    js = f'''
        <script>
            const element = window.parent.{js_path};
            element.scrollIntoView();
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
    
if st.sidebar.button('scroll to bottom of 1st container'):
    js_path = 'document.querySelector("#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-uf99v8.egzxvld5 > div.block-container.css-1y4p8pa.egzxvld4 > div:nth-child(1) > div > div:nth-child(2) > div > div:nth-child(1) > div > div > div > h2 > div > span")'
    scroll(js_path=js_path)
    
if st.sidebar.button('scroll to bottom of 2nd container'):
    js_path = 'document.querySelector("#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-uf99v8.egzxvld5 > div.block-container.css-1y4p8pa.egzxvld4 > div:nth-child(1) > div > div:nth-child(3) > div > div:nth-child(1) > div > div > div > h2 > div > span")'
    scroll(js_path=js_path)
    
if st.sidebar.button('scroll to bottom of main'):
    scroll(js_path='document.querySelector("footer")')