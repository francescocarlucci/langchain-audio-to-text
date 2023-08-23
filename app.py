import os
import openai
import tempfile
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#from langchain.docstore.document import Document

st.set_page_config(
    page_title="Convert Voice Memos to Text | Learn LangChain",
    page_icon="🔊"
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

with st.form("audio_text"):

	language = st.selectbox(
	'Output Language',
	('English', 'Italian'))

	voice_memos = st.file_uploader("Upload your voice memos", type=["m4a", "mp3"])

	execute = st.form_submit_button("🚀 Convert it")

	if execute:

		with st.spinner('Converting your voice memos...'):

			if voice_memos is not None:

				file_name, file_extension = os.path.splitext(voice_memos.name)

				with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temporary_file:

					temporary_file.write(voice_memos.read())

				audio_file= open(temporary_file.name, "rb")
				
				transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=openai_key)

				#doc = Document(
				#	page_content = transcript['text'],
				#	metadata = {}
				#)

				llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)

				prompt = ChatPromptTemplate.from_template('''
				Given the following transcript, please rephrase it as it was a small
				chapter of a book, make sure it is grammatically correct and make sure to include
				all the key concepts. Give also a title to the small chapter.
				Answer in {language} and use a friendly and simple tone.
				{transcript}
				''')

				chain = LLMChain(llm=llm, prompt=prompt)

				response = chain.run({
					'transcript': transcript,
					'language': language,
				})
				
				st.write(response)

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