import os
import openai
import tempfile
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

st.set_page_config(
    page_title="Convert Voice Memos to Text | Learn LangChain",
    page_icon="🔊"
)

# custom audio loade using OpenAI whisper API
def CustomAudioLoader(file, file_name, api_key):

	audio_file = open(file.name, "rb")
			
	transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=api_key)

	return Document(
		page_content = transcript['text'],
		metadata = { 'file_name': file_name }
	)

st.header('🔊 Convert Voice Memos to Text')

st.subheader('Learn LangChain | Demo Project #5')

st.success("This is a demo project related to the [Learn LangChain](https://learnlangchain.org/) mini-course.")

st.write('''
...
''')

st.info("You need your own keys to run commercial LLM models.\
    The form will process your keys safely and never store them anywhere.", icon="🔒")

openai_key = st.text_input("OpenAI Api Key", help="You need an account on OpenAI to generate a key: https://openai.com/blog/openai-api")

voice_memos = st.file_uploader("Upload your voice memos", type=["m4a", "mp3"])

post_processing = st.checkbox('Process your text transcript with a custom prompt')

with st.form("audio_text"):	

	if post_processing:

		custom_prompt = st.text_area("Custom prompt")

		st.write('''
		To further process your transcript effectively, the prompt should start with:
		"Given the following transcript...". Here are a few examples:
		- Given the following transcript, please change the tone of the voice and make it very formal.
		- Given the following transcript, please translate it to *
		- Given the following transcript, please summarize it in * words making sure the core concepts are included
		''')

	execute = st.form_submit_button("🖊️ Process Voice Memos")

	if execute:

		with st.spinner('Converting your voice memos...'):

			if voice_memos is not None:

				file_name, file_extension = os.path.splitext(voice_memos.name)

				with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temporary_file:

					temporary_file.write(voice_memos.read())

				audio_doc = CustomAudioLoader(temporary_file, file_name, openai_key)

				if post_processing:

					llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)

					prompt = ChatPromptTemplate.from_template('''
					{prompt}
					{transcript}
					''')

					chain = LLMChain(llm=llm, prompt=prompt)

					response = chain.run({
						'prompt': custom_prompt,
						'transcript': audio_doc.page_content
					})
					
					st.write(response)

				else:

					st.write(audio_doc.page_content)

				# clean-up the temporary file
				os.remove(temporary_file.name)

with st.expander("Exercise Tips"):
	st.write('''
	This demo is probably the most interesting one to expand and improve:
	- Browse [the code on GitHub](https://github.com/francescocarlucci/wordpress-code-assistant/blob/main/app.py) and make sure you understand it.
	- Fork the repository to customize the code.
	''')

st.divider()

st.write('A project by [Francesco Carlucci](https://francescocarlucci.com) - \
Need AI training / consulting? [Get in touch](mailto:info@francescocarlucci.com)')