import os
from pyngrok import ngrok
from threading import Thread

#Set Ngrok token
ngrok.set_auth_token("2eFqHIK3WMZdd1P7R0ZkWI4dxNI_4dtGYa9jiqHY85JW3F4PX")

#Open Tunnel to port 
url = ngrok.connect(addr = 8501,
                    proto = 'http',
                    bind_tls=True)

def run_streamlit():
    os.system('streamlit run "/scratch/users/k23058970/LLM-101/app.py" --server.port 8501')

print("App is live at:",url)

#Run app
thread = Thread(target=run_streamlit)
thread.start()