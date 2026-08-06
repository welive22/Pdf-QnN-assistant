# PDF Question Answering Assistant (RAG)

**Name:** EB Fathima Suhana
**MUID:** fathimasuhana@mulearn

**Deployment Link:** 

## Project Overview

This was for Assignment 11 of Epochs '26, the final one. The task was to build a PDF Question Answering app using RAG (Retrieval-Augmented Generation), so basically you upload a pdf, the app reads it, and you can ask it questions about the content and it answers based on whats actually in the document instead of just making stuff up. It also needed to remember earlier questions so you can ask natural follow-ups.

Built it with Streamlit for a clean chat-style interface, and deployed on Streamlit Community Cloud.

## Technologies Used

- **Streamlit** — for the interface (file upload + chatbot)

- **LangChain** — to wire everything together (loading, splitting, retrieval, memory, the chain itself)

- **PyPDFLoader** — to load and read the uploaded pdf

- **RecursiveCharacterTextSplitter** — splits the pdf text into overlapping chunks (1000 characters, 150 overlap) so the model gets manageable pieces of context instead of the whole document at once

- **Sentence Transformers** (`all-MiniLM-L6-v2`) — turns each chunk into an embedding vector, runs locally so no API dependency

- **ChromaDB** — stores the embeddings and does the similarity search to find relevant chunks for a given question

- **Groq API** (llama-3.3-70b-versatile, Free Tier) — the actual LLM that generates the answer, using the retrieved chunks as context

## Memory Implementation

Used LangChain's `ConversationBufferMemory` combined with `ConversationalRetrievalChain`. Every question and answer pair gets added to the memory buffer, and on the next question, the chain uses that history to understand what a follow-up question is actually referring to instead of treating every question as if its the first one. So something like asking "what is a stack" and then following up with "what about a queue, whats the difference" actually works, since it remembers what was just discussed.

Each uploaded PDF gets its own vector store (stored in a separate temp folder), so switching to a new PDF doesn't mix up content from a previous one, and clearing the chat resets the memory buffer without needing to re-process the whole PDF again.

## Challenges Faced

- Had version compatibility issues between different LangChain packages (langchain, langchain-community, langchain-huggingface) — some of the classes I first tried using like `ConversationBufferMemory` and `ConversationalRetrievalChain` are being phased out in the newest LangChain versions, so had to pin specific versions in requirements.txt that still support them properly instead of just using the latest of everything.

- Ran into dependency conflicts on Hugging Face Spaces with `huggingface-hub` versions clashing between `transformers`, `gradio`, and `chromadb`. Switched to Streamlit Community Cloud to avoid this entirely — Streamlit has a much simpler and more reliable dependency resolution on their platform.

- Getting the chunk size right took some trial — too small and the model loses context across chunks, too big and it wastes tokens / slows down retrieval. Went with 1000 characters with some overlap as a reasonable middle ground.

- Making sure a new PDF upload doesn't just add onto the previous one's vector store, so it needed a fresh vector store per upload instead of reusing the same one.

## Future Improvements

- Show which part/page of the PDF the answer actually came from (source citations), so the user can verify it themselves instead of just trusting the answer.

- Support uploading and querying multiple PDFs at once instead of just one at a time.

- Add a way to export the chat history for reviewing answers later.

- Try a better embedding model to see if retrieval quality improves for longer or more technical documents.

- Add basic handling for scanned/image-based PDFs (currently it only works on PDFs with actual selectable text).

- Try a better embedding model to see if retrieval quality improves for longer or more technical documents.

- Add basic handling for scanned/image-based PDFs (currently it only works on PDFs with actual selectable text).
