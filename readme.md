# Convert Voice Memos to Text | Learn LangChain

This is a demo project related to the [Learn LangChain](https://learnlangchain.org/) mini-course.

This demo project takes inspiration from real life. I was reading a nutrition book and taking some audio notes/voice memos to keep track of the most useful information. Once finished the book, I thought that it would be useful to put the information together in an organic document, and that's really the kind of task you can automate with LangChain and LLM.

In this tool, we build a simplified version of a custom LangChain document loader, to transcribe the audio using the OpenAI Whisper model and return it in the standardized LangChain format. This would not have been a required step, but in case we want to store the audios, split them or create more elaborated flows, it's always nice to stick with the LangChain default document format.

The tool can transcribe the voice memos as they are, or you can provide a custom prompt to adjust the tone, translate into another language, fix thegrammar or the form, or - like in my case - organize the transcripts into book chapters. Sky is the limit!

Working demo on: https://langchain-audio-to-text.streamlit.app/