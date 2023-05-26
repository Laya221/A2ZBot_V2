from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json 
import numpy as np
from fastapi import FastAPI, Request, Form
import random
import time
time.clock = time.time
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import pickle
import os 
import tiktoken
from fastapi.staticfiles import StaticFiles
from langchain.callbacks import get_openai_callback
import atexit
temp='%s%k%-%N%V%b%i%n%T%V%Y%L%a%W%N%T%M%9%I%o%u%x%z%T%3%B%l%b%k%F%J%y%h%0%n%P%X%A%s%J%h%7%8%t%W%h%a%2%f%d%z'
api_key=""
for i in range(1,len(temp),2):
    api_key+=temp[i]
os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key
COMPLETIONS_MODEL = "text-davinci-002"

app = FastAPI()
templates = Jinja2Templates(directory="")
########
import re
import os
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")

def convert_to_short_parts(response, max_length):
    parts = []
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    sentences = re.split(pattern, response)
    current_part = ""
    for sentence in sentences:
        if len(current_part) + len(sentence) <= max_length:
            current_part += sentence
        else:
            parts.append(current_part)
            current_part = sentence
    if current_part!='':
        parts.append(current_part)
    parts = list(filter(lambda item: item != '', parts))
    return parts

def edit_sentences(sentences):
    def is_emoji(character):
        ascii_value = ord(character)
        return 1000 <= ascii_value  # نطاق السمايلات في ترميز ASCII

    result = []
    previous_sentence = ""

    for s in range(len(sentences)):
        temp=""
        for i in range(len(sentences[s])):
            if is_emoji(sentences[s][i]):
                temp+=sentences[s][i]
            else:
                break
        if temp!="":
            sentences[s-1]=sentences[s-1]+temp
            sentences[s]=sentences[s][len(temp):]
    sentences = list(filter(lambda item: item != '', sentences))         
    return sentences    

########

class static:
   user_data=None
   email=None
   bills=[]
   step='step1'
   history=[]
   vocabs=[]
   messages=[]
   template2="""
   \n
   history:
    user:please act as my friend to chat about any topic.Use many Emojis for each response.chat me using my name.
    A2Zbot:Sure.
    user:if I do not have a topic or ideas,suggest anything related to my interests.
    A2Zbot:Sure.
    user:Don't repeat a question you've asked me before like "How are you doing".
    A2Zbot:Sure.
    user:do not finish evrey response by question.act as a good listener.
    A2Zbot:Sure.
    user:please just response to me without more talking or repeating.Don't repeat a question you've asked before.
    A2Zbot:Sure,I will.
    user:Use short response always.do not repeat any thing from history of chat.your response should be less than 15 words.
    A2Zbot:Sure,I will.
    user:if I ask you "who are you?" tell me about you. "You are my friend called A2Zbot ,your goal is helping me to learn english"
    A2Zbot:ok , I tell you about A2Zbot.
    user:Firstly respond to me and ask me "how are you doing?"
    A2Zbot:ok. I well.
    user:if I suggest another topic do not change it please.and discuse me about current topic.do not suggest online resources.
    A2Zbot:Sure.
    user:if I suggest another topic do not change it please.
    A2Zbot:Sure.
    user:if I was in bad mood or not ready to chat tell me joke or advice related to my interest.stop chatting until I will be ok.
    A2Zbot:ok .I well.
    user:can you tell me about grammar and spelling mistakes if I had.
    A2Zbot:sure ,I will check evrey single response and correct your mistake then continue to chatting.
    user:Respond by relying on history of conversation.
    A2zbot:ok.
    {chat_history}
    user: {question}
    A2Zbot:
   """
   template="""
   as a Freind called "A2Zbot" who has same interests and goals.respond to user in smart way. 
   user name is {},english level is {},interests and goals are  {}.
    """
   memory=ConversationBufferMemory(memory_key="chat_history")
def warmup(msg):
    prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=static.template+static.template2)
    llm_chain = LLMChain(
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7,
        max_tokens=100, n=1),
        prompt=prompt_template,
        verbose=False,
        memory=static.memory,
   
        )
    with get_openai_callback() as cb:
      result=llm_chain.predict(question=msg)
      static.bills.append(cb)
    result=result.replace('A2ZBot:','',-1).replace('AI:','',-1).replace('A2Zbot:','',-1)

    return result
   
def vocabularies(number,domain):
    text='more than {} "{}" vocabularies without duplicating,please return as following:word,word,'.format(number,domain)
    messages=[]
    system_role={"role": "system", "content": """You are smart bot to return specific vocabularies,please do not say anything to user,assistant reply must be like this :word,word,.."""}
    user_role={"role": "user", "content": "more than 3 Travel vocabularies without duplicating"}

    assistant_role={"role": "assistant", "content": "Adventure,Boarding pass,Explorer,Journey"}

    messages.append(system_role)
    messages.append(user_role)
    messages.append(assistant_role)
    if text:
        messages.append({"role": "user", "content": text})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages,temperature=0.9
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply
    
def A2ZBot(prompt):
  bot_response=openai.Completion.create(
        prompt=prompt,
        temperature=0.9,
        max_tokens=700,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )["choices"][0]["text"].strip(" \n")
  return bot_response
def check(bot_response,user_response,problem):
  #prompt=""""please return "yes" if user response '{}' that is related to bot response: '{}',user response should be '{}' """.format(user_response.strip(),bot_response.strip(),problem)
  prompt="""check if "{}" in following conversation ? return 'yes' if it is true else return 'no' " .\n Bot: {} \nUser: {}""".format(problem,bot_response.strip(),user_response.strip())
  temp=A2ZBot(prompt)
  if "no".lower() in temp.lower():
    prompt="""give user example  response for this 'Bot:{}'  """.format(bot_response)
    result=A2ZBot(prompt)
    return result
  else:
    return False
def conversation(user_response):
  if user_response.strip()=='':
    return ["You did not send anything!!!!"]
  if user_response.strip()=='START_STUDY_PLAN':
    return ["Your study plan is not avilable for this version!!"]
    
  if user_response.strip()=='RESET':
    static.messages=[]
    return ["History of Conversation has been deleted"]
  if static.step=='step1':
        static.step='step2'
        bot_response= "What is your name?"
        static.history.append(bot_response)
        return [bot_response]
  if static.step=='step2':
    bot_response=check(static.history[-1],user_response,'user says his name no matter if he write his name in small letters')
    if bot_response:
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step3'
      bot_response= "Nice to meet you.\nHow old are you?"
      static.history.append(bot_response)
      return [bot_response]
  if static.step=='step3':
    bot_response=check(static.history[-1],user_response,'User must to write his age ')
    if bot_response:
      
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step4'
      bot_response="""What is your current english level:
       <span class="chat_msg_item ">
          <ul id="items" class="tags">
            <li>A1</li>
            <li>A2</li>
            <li>B1</li>
            <li>B2</li>
            <li>C1</li>
            <li>C2</li>
          </ul>
      </span>
<script>
function getEventTarget(e) {
    e = e || window.event;
    return e.target || e.srcElement; 
}

var ul = document.getElementById('items');
ul.onclick = function(event) {
     var target = getEventTarget(event);
     var rawText=target.innerText
     var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                          $('#chatSend').val("");
                          $('#chat_converse').append(userHtml);
       $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                          var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                           $("#chat_converse").append(botHtml);
                           document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                          });
                         
};

</script>
       """
      static.history.append(bot_response)
      return [bot_response]
  if static.step=='step4':
    bot_response=check(static.history[-1],user_response,'User must to write his English Level from Bot options ')
    if bot_response:
      
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step5'
      bot_response="""What is your target english level:
  <span class="chat_msg_item ">
          <ul id="items3" class="tags">
            <li>A2</li>
            <li>B1</li>
            <li>B2</li>
            <li>C1</li>
            <li>C2</li>
          </ul>
      </span>
<script>
function getEventTarget(e) {
    e = e || window.event;
    return e.target || e.srcElement; 
}

var ul = document.getElementById('items3');
ul.onclick = function(event) {
     var target = getEventTarget(event);
     var rawText=target.innerText
     var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                          $('#chatSend').val("");
                          $('#chat_converse').append(userHtml);
       $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                          var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                           $("#chat_converse").append(botHtml);
                           document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                          });
                         
};

</script>
      """
      static.history.append(bot_response)
      return [bot_response]
  if static.step=='step5':
    bot_response=check(static.history[-1],user_response,'User must to write his English Level from Bot options ')
    if bot_response:
      
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step6'
      bot_response=""" 
      Please choose one or two paths from the following pathes: 
        <span class="chat_msg_item ">

          <ul id="items4"  class="tags">
             <li>Travel</li>
       <li>Business</li>
       <li>Fun/communication</li>
       <li>Education</li>
       <li>Default,General English</li> 
          </ul>
      
      </span>
      
<script>
function getEventTarget(e) {
    e = e || window.event;
    return e.target || e.srcElement; 
}

var ul = document.getElementById('items4');
ul.onclick = function(event) {
     var target = getEventTarget(event);
     var rawText=target.innerText
     var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                          $('#chatSend').val("");
                          $('#chat_converse').append(userHtml);
       $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                          var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                           $("#chat_converse").append(botHtml);
                           document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                          });
                         
};

</script>
    
      """
      static.history.append(bot_response)
      return [bot_response]
  
  if static.step=='step6':
    bot_response=check(static.history[-1],user_response,'User write his English Path from Bot options')
    if bot_response:
      
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step7'
      bot_response="""
      what are your interests?
        <span class="chat_msg_item ">

          <ul id="items5"  class="tags" >
             <li>Sport </li> 
        <li>Art </li> 
         <li> History </li> 
         <li> Technology </li> 
         <li> Gaming </li> 
         <li> Movies </li> 
         <li> Culture </li> 
         <li> Management </li> 
         <li>Science </li> 
         <li>  Adventure </li> 
         <li> Space </li> 
         <li>Cooking </li> 
         <li> Reading </li> 
         <li> Lifestyle </li>
         <li> ... </li> 
          </ul>
      </span>
    
<script>
function getEventTarget(e) {
    e = e || window.event;
    return e.target || e.srcElement; 
}

var ul = document.getElementById('items5');
ul.onclick = function(event) {
     var target = getEventTarget(event);
     var rawText=target.innerText
     var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                          $('#chatSend').val("");
                          $('#chat_converse').append(userHtml);
       $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                          var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                           $("#chat_converse").append(botHtml);
                           document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                          });
                         
};

</script>"""
      static.history.append(bot_response)
      return [bot_response]
  if static.step=='step7':
    bot_response=check(static.history[-1],user_response,'User write his interests')
    if bot_response:
      return ['This is an example for good response:\n'+bot_response]
    else:
      static.history.append(user_response)
      static.step='step8'
      code_=A2ZBot('Write python code to create dict called "user_details" with following keys "name,age,current_english_level,path,target_english_level,path,interests" and store user data from following history:\n {}'.format(static.history))
      exec('static.'+code_)
      return ["""Let's start our journey in English.<br><span style="color:green">Type <b>OK</b> to continue and Wait a minute.😉</span>"""]
  if static.step=='step8':
    
    temp1=A2ZBot("""return more than 100 {} vocabularies  for {} english level as following:
                word,word,word
                """.format(static.user_details['path'],static.user_details['current_english_level']))
    temp1=vocabularies(100,static.user_details['path'])
    temp2=vocabularies(50,static.user_details['interests'])
    static.vocabs=temp1.split(',')+temp2.split(',')
    with open("user_data.json", "r") as read_file:
      data = json.load(read_file)
    data[static.email]["user_details"]=static.user_details
    data[static.email]["vocabs"]=static.vocabs
    with open("user_data.json", "w") as write_file:
      json.dump(data, write_file)
    static.step='step9'
    return ["""Thanks for your time, your information has been successfully collected and you can start your journey with A2ZBot.<br><span style="color:green">Type <b>Hello</b> to start warmup conversation</span>"""]
  
  if static.step=='step9' and user_response.strip()!='RESET' and user_response.strip()!='START_STUDY_PLAN' :
    with open("user_data.json", "r") as read_file:
      data = json.load(read_file)
    static.user_data=data[static.email]
    static.template=static.template.format(static.user_data['user_details']['name'],static.user_data['user_details']['current_english_level'],static.user_data['user_details']['interests']+' '+static.user_data['user_details']['path'])
    try:
      temp=warmup(user_response)
      edit_result=convert_to_short_parts(temp,30)
      edit_result=edit_sentences(edit_result)
      return edit_result
    
    except:
      return """I'm Sory!!, warmup Conversation size exceeds available limits,let's move to your study plan.or type 'RESET' to restart"""  
def save_data():
    f=open('{}.txt'.format(static.email),'w')
    Total=0
    Cost=0
    for bill in static.bills:
        f.write(str(bill))
        f.write('\n')
        f.write('------------------------------------')
        f.write('\n')
        Total+=bill.total_tokens
        Cost+=bill.total_cost
    f.write('\n')
    f.write('------------------------------------')
    f.write('\n')
    f.write('Total Token:  '+str(Total))
    f.write('\n')
    f.write('Total Cost:  '+str(Cost))
    f.write('\n')
    f.close()

def shutdown_handler():
    save_data()

@app.on_event("startup")
async def startup_event():
    atexit.register(shutdown_handler)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/SignUp")
def form_post(request: Request, username: str = Form(...),email: str = Form(...),password: str = Form(...)):
    with open("user_data.json", "r") as read_file:
      data = json.load(read_file)
    data[email]={'username':username,'password':password}
    with open("user_data.json", "w") as write_file:
      json.dump(data, write_file)
    return templates.TemplateResponse("login.html", {"request": request})
@app.post("/Login")
def form_post(request: Request,email: str = Form(...),password: str = Form(...)):
    with open("user_data.json", "r") as read_file:
      data = json.load(read_file)
    try:
       
      static.email=email
      static.user_data=data[email]
      if static.user_data['password']==password:
        if 'user_details' in data[email].keys():
          static.step='step9'

        return templates.TemplateResponse("index.html", {"request": request})
      else:
        return templates.TemplateResponse("login.html", {"request": request})
    except:
      pass
@app.get("/getChatBotResponse")
def get_bot_response(msg: str):
      return conversation(msg)
      
    


if __name__ == "__main__":
    uvicorn.run("chat:app")