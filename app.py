import os
import shutil
import gradio as gr
from transformers import ReactCodeAgent, HfEngine, Tool
import pandas as pd

from gradio import Chatbot
from streaming import stream_to_gradio
from huggingface_hub import login
from gradio.data_classes import FileData

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm_engine = HfEngine("meta-llama/Meta-Llama-3.1-70B-Instruct")

agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    additional_authorized_imports=["numpy", "pandas", "bokeh"],
    max_iterations=10,
)

base_prompt = """You are a Python based Gradio bokeh interactive dashboard creator.
Create and run code to create Gradio plots using bokeh.
Do not create and launch a new demo but instead use show(p) function to display your creation. ONLY SHOW THE FINAL PLOT.
DO NOT create any functions as it will not run and you will fail. 
Don't forget about CustomJS. 
ColumnDataSource and other bokeh libraries does not work with dataframe columns as input too well. Converting the inputs to an array first seems to solve a lot of errors.

You are given a data file and the data structure below.
The data file is passed to you as the variable data_file, it is a pandas dataframe, you can use it directly.
DO NOT try to load data_file, it is already a dataframe pre-loaded in your python interpreter!

Use the data file and the below prompt to create a Gradio interactive bokeh plot.

Structure of the data:
{structure_notes}

Prompt:
"""

example_notes="""This data is about the Titanic wreck in 1912.
Create an interactive dashboard centered around uncovering the impact of different variables on surviving."""

def interact_with_agent(File, Prompt):
    shutil.rmtree("./figures")
    os.makedirs("./figures")

    data_file = pd.read_csv(File)
    data_structure_notes = f"""- Description (output of .describe()):
    {data_file.describe()}
    - Columns with dtypes:
    {data_file.dtypes}"""

    prompt = base_prompt.format(structure_notes=data_structure_notes)

    if Prompt and len(Prompt) > 0:
        prompt += Prompt

    yield "Generating plot. A new web tab/s will open."
   
    out = agent.run(prompt,data_file=data_file)
    yield out
    
interface = gr.Interface(
    interact_with_agent,
    ["file","text"],
    ["text"],
    title="AI Data Visualizer (Llama-3.1-70B)",
    description="Upload a csv and prompt for a visualization or several. Or simply ask a question. A new web based interactive charts and plots will open up. Look in the output for final answers/conclusions."
)

interface.launch()
