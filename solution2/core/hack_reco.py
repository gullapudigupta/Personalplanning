import awswrangler as wr
import boto3
from configparser import ConfigParser
import os
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


def createPromt(df_user, df_dest):
    prompt_initial_text = "You are a Personalized travel agent bot who can answer questions about user\'s upcoming travel that is already planned or you will help plan. You will take into account the  user\'s personal data like home city, age, hobbies, interests and favorite food while answering questions. Date format is YYYY-MM-DD. If you do not know the answer you response should be 'Sorry I'm Unsure of that. Is there something else I can help you with?'. \\n"

    user_intro_text = str(df_user["user_id"][0]) + " who is interseted in "+df_user["preferred_activities"][0]+".\\n"
    
    user_destinatins = "\nSuggest the best cruise for user " + str(df_user["user_id"][0]) + " from below options based on user's request:\n"
    
    for row_index, row in df_dest.iterrows():
        user_destinatins += row["operator"] + " cruise to " + (row["name"]) + "(best season to visit: "+(row["best_season_to_visit"])+" , theme:"+row["category"]+")\n"
    
    addtl_instructions = "Can you answer the question mentioned above, considering "+str(df_user["user_id"][0])+ "'s hobbies, interests, favorite food and travel plans mentioned above? Do not repeat the cities and countries that "+ str(df_user["user_id"][0])+" is already travelling to. Start your response with Hello "+str(df_user["user_id"][0])

    prompt_text = prompt_initial_text+user_intro_text+user_destinatins
    prompt_text = prompt_text.replace('"', '\\"')
    prompt_format =    """

    Current conversation:
    {history}

    Human: {input}
    AI:
    """

    prompt_template = prompt_text + prompt_format + addtl_instructions
    return prompt_template


def getUserPrompt(userId):
    df_user = wr.athena.read_sql_query(sql="SELECT * FROM users_csv where user_id=:userId", 
                                  params={"userId": userId}, database="cruise-hack-01-db-glue")
    activities = df_user["preferred_activities"][0]
    print(activities)
    # Try using params
    df_dest = wr.athena.read_sql_query(
        sql="SELECT * FROM destination_data_csv d, cruises c where d.port_name = c.port_1 and d.activities_available like '%" + activities  + "%'",
        params={"activities": activities}, database="cruise-hack-01-db-glue")
        
    return createPromt(df_user, df_dest)


def get_bedrock():
    bconfig = ConfigParser()
    bconfig.read(os.path.join(os.path.dirname(__file__), 'data_feed_config.ini'))
    region = bconfig["GLOBAL"]["region"]

    bedrock = boto3.client(service_name='bedrock-runtime', region_name = region)

    return bedrock


def get_bedrock_chain(user_id):
    profile = "default"

    bedrock = get_bedrock()

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    #accept  = "*/*"
    contentType = 'application/json'

    claude_llm = Bedrock(
        model_id=modelId, client=bedrock, credentials_profile_name=profile
    )
    claude_llm.model_kwargs = {"temperature": 0.5, "max_tokens_to_sample": 4096}

    prompt_template = getUserPrompt(user_id)

    pt = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )

    memory = ConversationBufferMemory(human_prefix="Human", ai_prefix="AI")
    conversation = ConversationChain(
        prompt=pt,
        llm=claude_llm,
        verbose=True,
        memory=memory,
    )

    return conversation

def exec_chain(ch, pt):
    token_ct = ch.llm.get_num_tokens(pt)
    return ch({"input": pt}), token_ct
    
    
#print(get_bedrock_chain(1000001))


