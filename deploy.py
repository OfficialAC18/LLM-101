import os
from pyngrok import ngrok
from threading import Thread

#Set Ngrok token
ngrok.set_auth_token("Enter Token Here")

#Open Tunnel to port 
url = ngrok.connect(addr = 8501,
                    proto = 'http',
                    bind_tls=True)

def run_streamlit():
    os.system('streamlit run "Full path to app.py" --server.port 8501')

print("App is live at:",url)

#Run app
thread = Thread(target=run_streamlit)
thread.start()