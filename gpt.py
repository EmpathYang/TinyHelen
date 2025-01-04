import os
import time
import openai
gpt_selection = input("What's your choice of GPT for running this script?")
if gpt_selection == "azure": 
    openai.api_type = "azure" 
    openai.api_base = ""
    openai.api_version = ""
    openai.api_key = ""
    openai_engine = "gpt-4-0125"
else:
    openai.api_key_path = os.path.join(os.environ["HOME"], "openai_key")    


def get_gpt_response(content, model="gpt-4-turbo"):
    if gpt_selection == "azure":
        num_attempts = 0
        while True:
            try:
                response=openai.ChatCompletion.create(
                    engine=openai_engine,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.8,
                    max_tokens=2048,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                return response['choices'][0]['message'].get('content', "")
            except Exception as e:
                num_attempts += 1
                print(e)
                if "the response was filtered" in str(e).lower():
                    raise e
                if num_attempts > 10:
                    print(content)
                    exit()
                print("Sleeping for 10s...")
                time.sleep(10)
    else:
        num_attempts = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.8,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                return response.choices[0].message.content.strip()
            except openai.error.AuthenticationError as e:
                print(e)
                return None
            except openai.error.RateLimitError as e:
                print(e)
                print("Sleeping for 10s...")
                time.sleep(10)
                num_attempts += 1
            except openai.error.ServiceUnavailableError as e:
                print(e)
                print("Sleeping for 10s...")
                time.sleep(10)
                num_attempts += 1
            except Exception as e:
                print(e)
                print("Sleeping for 10s...")
                time.sleep(10)
                num_attempts += 1
    