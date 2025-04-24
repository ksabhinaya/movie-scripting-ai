import google.generativeai as genai

genai.configure(api_key='AIzaSyBxLF3K4RKbnw0HFqD0J_bud0rIBzPExok ')

try:
    models = genai.list_models()
    for model in models:
        print(model.name)
except Exception as e:
    print("Error:", e)