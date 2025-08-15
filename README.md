# RAG LLM Project

This project provides a web-based and command-line interface for a Retrieval-Augmented Generation (RAG) system. You can index local code repositories or document folders and ask questions about them using either OpenAI or a local Ollama model.

## Features
-   Web UI for indexing, querying, and configuring the application.
-   Command-line interface for scripting and terminal-based usage.
-   Supports both OpenAI and local Ollama LLMs.
-   Uses ChromaDB for persistent vector storage.
-   Includes an optional re-ranking step to improve retrieval quality.

## Getting Started

### 1. Prerequisites
-   Python 3.8+
-   An OpenAI API key (if using OpenAI models).
-   [Ollama](https://ollama.com/) installed and running (if using local models like `llama2` or `mistral`).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rag_llm
    ```

2.  **Install Dependencies:**
    Install the required Python packages. It's highly recommended to do this in a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install packages from requirements.txt
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, you will need to install the packages imported in the project, such as `fastapi`, `uvicorn`, `python-multipart`, `llama-index`, `chromadb`, etc.)*

### 3. Configuration

1.  The application uses a `config.yaml` file in the root directory. You can create one by copying `config.example.yaml` if it exists, or by creating a new file.
2.  **Create `config.yaml`:** Before running the application for the first time, you must create your own configuration file from the provided template.
    ```bash
    cp config.example.yaml config.yaml
    ```
3.  **Edit `config.yaml`:** Open the newly created `config.yaml` and customize the settings:
    *   **`storage_path`**: This is a crucial setting that defines where your vector database will be stored. The default is `./storage`.
    *   **`llm.api_key`**: If you are using OpenAI, you must add your API key here.
    *   **`project_paths`**: You can pre-populate the list of directories you want to index for easy access in the UI.

#### Embedding Model Configuration

The `embedding` section in the `config.yaml` file is critical for the correct functioning of the RAG system. It specifies the model used to generate embeddings for both the documents during indexing and the queries during retrieval. It is crucial that the same embedding model is used for both processes.

*   **`embedding.model`**: This setting defines the Hugging Face model to be used for generating embeddings. The default model is `sentence-transformers/all-MiniLM-L6-v2`. You can change this to any other sentence-transformer model from the Hugging Face Hub.

**Important:** If you change the embedding model after you have already indexed documents, you will need to re-index your documents for the changes to take effect. This is because the new model will generate embeddings with a different dimension, which will cause a mismatch with the existing embeddings in the vector store.

### 4. How to Run

You can interact with the application through the Web UI or the Command-Line Interface.

#### Running the Web Application (Backend + Frontend)

1.  **Start the Backend Server:**
    From the project's root directory, run the following command to start the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`.

2.  **Access the Frontend:**
    Open your web browser and navigate to `http://127.0.0.1:8000`. From here you can index directories and chat with your documents.

#### Using the Command-Line Interface (CLI)

1.  **To Index a Project:**
    ```bash
    python cli.py index --path /path/to/your/project
    ```
2.  **To Ask a Question:**
    ```bash
    python cli.py query "What is the main purpose of the indexer script?"
    ```

---

### Project Architecture and Pipeline

This section provides an overview of the project's components and the data flow for both indexing and querying.

#### Core Components

*   **FastAPI Backend (`app/main.py`):** Serves the web interface and provides API endpoints for indexing (`/index`), querying (`/query`), and managing settings (`/config`).
*   **Web Interface (HTML/JS/CSS):** A single-page application in `app/templates` and `app/static` that allows users to interact with the backend.
*   **Command-Line Interface (`cli.py`):** Offers a terminal-based way to run the indexing and querying processes.
*   **Indexer (`app/core/indexer.py`):** Responsible for reading source documents, generating embeddings, and storing them.
*   **Querier (`app/core/querier.py`):** Handles user questions by retrieving relevant documents from the index and using an LLM to generate an answer.
*   **Configuration (`config.yaml`):** A central file to manage all settings, such as API keys, model names, and storage paths.

#### Indexing Pipeline

When you index a directory, the following steps occur:

1.  **Initiation:** The user provides an absolute path to a directory via the Web UI or the CLI.
2.  **Document Loading:** `SimpleDirectoryReader` from LlamaIndex recursively scans the directory and loads all supported files (e.g., `.py`, `.md`, `.txt`) into memory.
3.  **Embedding Model Setup:** The Hugging Face `sentence-transformers` model specified in `config.yaml` is loaded to convert text into numerical vectors (embeddings).
4.  **Vector Store Setup:** A persistent `ChromaDB` client is initialized at the `storage_path` defined in the configuration. A collection is created to hold the vectors.
5.  **Indexing:** LlamaIndex's `VectorStoreIndex` takes the loaded documents, chunks them into smaller pieces, uses the embedding model to create a vector for each chunk, and stores these vectors along with their corresponding text in the ChromaDB collection.
6.  **Metadata Persistence:** LlamaIndex's own metadata (document store, index store) is saved to the same storage path to allow for easy reloading of the index later.

#### Querying Pipeline

When you ask a question, the RAG (Retrieval-Augmented Generation) pipeline is executed:

1.  **Initiation:** The user submits a question through the Web UI or CLI.
2.  **Configuration Loading:** The system loads the index from the `storage_path`. It configures the LLM (either `Ollama` or `OpenAI`) and the `HuggingFaceEmbedding` model based on the user's choice and the `config.yaml` file. It is crucial that the same embedding model is used for both indexing and querying.
3.  **Query Engine Creation:** A LlamaIndex `QueryEngine` is created from the loaded index.
4.  **Retrieval:**
    a. The user's question is converted into a vector using the same embedding model from the indexing phase.
    b. The system performs a similarity search in ChromaDB to find the document chunks with vectors most similar to the question's vector. The number of chunks retrieved is determined by `similarity_top_k`.
5.  **Re-ranking (Optional):** If enabled, a `SentenceTransformerRerank` model takes the retrieved chunks and re-scores them based on their relevance to the original question, providing a more refined context.
6.  **Generation (Augmentation):** The original question, along with the retrieved (and re-ranked) document chunks, are combined into a single prompt. This is sent to the selected LLM (OpenAI or Ollama).
7.  **Response:** The LLM generates an answer based on the provided context. The final answer and the source document chunks used to generate it are sent back to the user.

---

### Project Enhancements Log

#### 4.9.4 Configuration and Stability Refactor

**Enhancement:** The application's configuration management has been centralized and made more robust, fixing critical bugs related to inconsistent state between the web UI and the command line.

**Implementation Details:**
*   **Centralized Configuration:** The `indexer`, `querier`, and `cli` modules no longer load their own configuration files. Instead, they receive the active configuration object as a function argument. This ensures all parts of the application use the same, up-to-date settings.
*   **Safe Configuration Updates:** The `/config` endpoint in `app/main.py` was fixed to perform a deep merge of settings. This prevents accidental deletion of keys (like `storage_path`) when updating settings from the web UI.
*   **Configuration Template (`config.example.yaml`):** A new `config.example.yaml` file was created to serve as a clear and documented template for users, ensuring all necessary keys are present.
*   **Consistent Storage Path:** All parts of the application now correctly use the `storage_path` defined in `config.yaml`, preventing the creation of default `data/` directories.

---

#### 4.9.3 Enhanced Application Settings UI

**Enhancement:** The settings modal now provides a more user-friendly experience with appropriate input types and helpful descriptions.

**Implementation Details:**
*   **`app/templates/index.html`:**
    *   Numerical settings with ranges (`llmTemperature`, `chunkSize`, `similarityTopK`, `rerankTopN`) were changed to `type="range"` sliders with `min`, `max`, and `step` attributes. `<span>` elements were added to display the current slider value.
    *   Help descriptions were added as `data-bs-toggle="tooltip"` attributes, providing brief explanations when hovering over the input fields.
*   **`app/static/script.js`:**
    *   The `fetchAndPopulateSettings()` function was updated to correctly populate values for range inputs and their associated `<span>` displays.
    *   Event listeners were added to slider inputs to dynamically update the displayed value as the slider is moved.
    *   Bootstrap tooltips were initialized for all elements with `data-bs-toggle="tooltip"`.
    *   The `settingsForm` submission logic was adjusted to correctly read values from the new input types (parsing as `parseFloat` or `parseInt` where appropriate).
*   **`app/static/style.css`:** Basic styling was added for the range input value displays.

---

### 5. Remaining Potential Enhancements (Future Work)

This section outlines further improvements that can be made to the project in terms of UI/UX, robustness, and advanced document handling.

#### 5.1 User Interface (UI/UX) Enhancements
*   **Real-time Indexing Progress:** Provide live feedback on the indexing process (e.g., a progress bar, number of files processed, current file being indexed). This would require backend updates to stream progress to the frontend.
*   **Markdown Rendering:** Display LLM responses with proper Markdown formatting (bold, italics, lists, code blocks).
*   **Enhanced Loading Indicators:** More prominent or animated loading indicators while the LLM is generating a response.
*   **More Detailed Error Messages:** Display more user-friendly and informative error messages on the frontend, rather than generic "Error" messages.
*   **File Upload (Limited):** Allow users to upload individual files (e.g., PDF, DOCX) via a file input, with the backend handling temporary storage and indexing. (Note: Direct folder selection from web UI is not feasible due to security restrictions).

#### 5.2 Stability and Robustness
*   **Comprehensive Error Handling:** Implement more granular `try-except` blocks in `indexer.py` and `querier.py` to catch specific exceptions (e.g., file access errors, LLM API errors, ChromaDB issues) and provide meaningful error messages.
*   **Asynchronous Operations (Backend):**
    *   **Indexing as a Background Task:** For long-running indexing operations, use FastAPI's `BackgroundTasks` or integrate a dedicated task queue (like Celery with Redis/RabbitMQ) to prevent the UI from freezing and allow the server to remain responsive.
    *   **Non-blocking LLM Calls:** Ensure LLM calls are truly asynchronous to prevent blocking the event loop.
*   **Structured Logging:** Implement a proper logging setup using Python's `logging` module. Log key events, warnings, and errors with relevant context (timestamps, module, function, error details).
*   **Configuration Validation:** Implement schema validation for `config.yaml` (e.g., using Pydantic or a simple schema checker) to ensure configuration values are valid before the application starts.
*   **Resource Management & Monitoring:** For local LLMs, consider adding basic checks for system resources (RAM, CPU) to warn users if their system might struggle with a particular model. Explore simple monitoring tools or metrics to track application performance (e.g., query latency, indexing time).
*   **Containerization (Docker):** Provide a `Dockerfile` and `docker-compose.yml` to containerize the application. This ensures consistent environments, simplifies deployment, and isolates dependencies.
*   **Unit and Integration Tests:** Write automated tests for core logic (indexing, querying, config loading) and API endpoints to ensure correctness and prevent regressions.

#### 5.3 Advanced Document Handling (True Incremental Indexing)
*   **Robust Change Detection:** Implement a system to accurately detect new, modified, and deleted files in the source directory. This would involve:
    *   **Metadata Storage:** Storing file metadata (e.g., file path, last modified timestamp, or a content hash) alongside the document in ChromaDB or a separate manifest.
    *   **Comparison Logic:** Before indexing, compare the current file system state with the stored metadata.
        *   **New Files:** Add to the ChromaDB collection and LlamaIndex.
        *   **Modified Files:** Delete the old document from ChromaDB and LlamaIndex, then add the new version.
        *   **Deleted Files:** Remove the corresponding document from ChromaDB and LlamaIndex.
*   **LlamaIndex API for Updates:** Leverage LlamaIndex's `delete_ref_doc` and `insert_nodes` methods on the `VectorStoreIndex` to precisely manage document lifecycle within the index.

---