# Author @ Misbah Iradat
import uvicorn
from fastapi import FastAPI
import REDEye_detection as live_predict


app = FastAPI()

@app.get('/')
async def project():
    return 'Red Eye Detection'

@app.post('/app/REDEye_detection')
async def predict_call():

    pred = live_predict
    return pred
   
# uvicorn app:app --reload
if __name__ =="__main__":

    uvicorn.run(app,port =8000, host ='0.0.0.0')

