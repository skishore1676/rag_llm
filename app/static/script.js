document.getElementById('predefinedPath').addEventListener('change', (event) => {
    document.getElementById('projectPath').value = event.target.value;
});

document.getElementById('indexForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const projectPath = document.getElementById('projectPath').value;
    const indexStatus = document.getElementById('indexStatus');
    indexStatus.textContent = 'Indexing...';
    indexStatus.style.backgroundColor = '#ffc107';

    try {
        const response = await fetch('/index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `project_path=${encodeURIComponent(projectPath)}`,
        });
        const data = await response.json();
        indexStatus.textContent = data.message;
        if (data.status === 'success') {
            indexStatus.style.backgroundColor = '#d4edda';
            indexStatus.style.color = '#155724';
        } else {
            indexStatus.style.backgroundColor = '#f8d7da';
            indexStatus.style.color = '#721c24';
        }
    } catch (error) {
        indexStatus.textContent = `Error: ${error.message}`;
        indexStatus.style.backgroundColor = '#f8d7da';
        indexStatus.style.color = '#721c24';
    }
});

let chatHistory = [];

function displayMessage(role, content) {
    const chatHistoryDiv = document.getElementById('chatHistory');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', role);
    messageElement.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> ${content}`;
    chatHistoryDiv.appendChild(messageElement);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; // Scroll to bottom
}

// Function to fetch and populate settings
async function fetchAndPopulateSettings() {
    try {
        const response = await fetch('/config');
        const config = await response.json();

        // Correctly select the model based on the LLM type
        const activeModel = config.llm.type === 'ollama' ? config.llm.ollama_model : config.llm.model;
        document.getElementById('llmModel').value = activeModel;
        document.getElementById('llmTemperature').value = config.llm.temperature;
        document.getElementById('llmTemperatureValue').textContent = config.llm.temperature;
        document.getElementById('chunkSize').value = config.indexing.chunk_size;
        document.getElementById('chunkSizeValue').textContent = config.indexing.chunk_size;
        document.getElementById('similarityTopK').value = config.retrieval.similarity_top_k;
        document.getElementById('similarityTopKValue').textContent = config.retrieval.similarity_top_k;
        document.getElementById('rerankEnableSettings').checked = config.rerank.enable;
        document.getElementById('rerankModel').value = config.rerank.model;
        document.getElementById('rerankTopN').value = config.rerank.top_n;
        document.getElementById('rerankTopNValue').textContent = config.rerank.top_n;

        // Initialize tooltips
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))

    } catch (error) {
        console.error('Error fetching config:', error);
        const settingsStatus = document.getElementById('settingsStatus');
        settingsStatus.textContent = `Error loading settings: ${error.message}`;
        settingsStatus.classList.add('alert-danger');
    }
}

// Update slider value display
document.getElementById('llmTemperature').addEventListener('input', (event) => {
    document.getElementById('llmTemperatureValue').textContent = event.target.value;
});
document.getElementById('chunkSize').addEventListener('input', (event) => {
    document.getElementById('chunkSizeValue').textContent = event.target.value;
});
document.getElementById('similarityTopK').addEventListener('input', (event) => {
    document.getElementById('similarityTopKValue').textContent = event.target.value;
});
document.getElementById('rerankTopN').addEventListener('input', (event) => {
    document.getElementById('rerankTopNValue').textContent = event.target.value;
});

// Event listener for settings form submission
document.getElementById('settingsForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const settingsStatus = document.getElementById('settingsStatus');
    settingsStatus.textContent = 'Saving settings...';
    settingsStatus.classList.remove('alert-success', 'alert-danger');
    settingsStatus.classList.add('alert-info');

    const selectedModel = document.getElementById('llmModel').value;
    const isOllamaModel = OLLAMA_MODELS.includes(selectedModel);
    const newConfig = {
        llm: {
            temperature: parseFloat(document.getElementById('llmTemperature').value),
        },
        indexing: {
            chunk_size: parseInt(document.getElementById('chunkSize').value),
        },
        retrieval: {
            similarity_top_k: parseInt(document.getElementById('similarityTopK').value),
        },
        rerank: {
            enable: document.getElementById('rerankEnableSettings').checked,
            model: document.getElementById('rerankModel').value,
            top_n: parseInt(document.getElementById('rerankTopN').value),
        }
    };

    // Set the correct model key based on whether it's an Ollama model or not
    if (isOllamaModel) {
        newConfig.llm.ollama_model = selectedModel;
    } else {
        newConfig.llm.model = selectedModel;
    }

    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(newConfig),
        });
        const data = await response.json();
        settingsStatus.textContent = data.message;
        settingsStatus.classList.remove('alert-info');
        settingsStatus.classList.add('alert-success');
        // Optionally, refresh the page or update other UI elements that depend on config
    } catch (error) {
        settingsStatus.textContent = `Error saving settings: ${error.message}`;
        settingsStatus.classList.remove('alert-info');
        settingsStatus.classList.add('alert-danger');
    }
});

// Fetch settings when the modal is shown
const settingsModal = document.getElementById('settingsModal');
settingsModal.addEventListener('show.bs.modal', fetchAndPopulateSettings);

document.getElementById('clearChatBtn').addEventListener('click', () => {
    chatHistory = [];
    document.getElementById('chatHistory').innerHTML = '';
    document.getElementById('queryAnswer').textContent = '';
    document.getElementById('queryAnswer').classList.remove('alert-success', 'alert-danger', 'alert-info');
});

document.getElementById('queryForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const questionInput = document.getElementById('question');
    const question = questionInput.value;
    const llmType = document.getElementById('llmType').value;
    const rerankEnable = document.getElementById('rerankEnable').checked;
    const queryAnswer = document.getElementById('queryAnswer');

    if (!question) return;

    displayMessage('user', question);
    chatHistory.push({ role: 'user', content: question });
    questionInput.value = ''; // Clear input

    queryAnswer.textContent = 'Searching...';
    queryAnswer.classList.remove('alert-success', 'alert-danger');
    queryAnswer.classList.add('alert-info');

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `question=${encodeURIComponent(question)}&llm_type=${encodeURIComponent(llmType)}&rerank_enable=${rerankEnable}&chat_history=${encodeURIComponent(JSON.stringify(chatHistory))}`,
        });
        const data = await response.json();
        
        displayMessage('assistant', data.answer);
        chatHistory.push({ role: 'assistant', content: data.answer });

        // Display source documents
        const sourceDocumentsDiv = document.getElementById('sourceDocuments');
        const sourceList = document.getElementById('sourceList');
        sourceList.innerHTML = ''; // Clear previous sources

        if (data.sources && data.sources.length > 0) {
            sourceDocumentsDiv.style.display = 'block';
            data.sources.forEach(source => {
                const listItem = document.createElement('li');
                listItem.classList.add('list-group-item');
                listItem.innerHTML = `<strong>Score:</strong> ${source.score.toFixed(4)}<br><strong>File:</strong> ${source.metadata['file_name']}<br><strong>Text:</strong> ${source.text.substring(0, 200)}...`;
                sourceList.appendChild(listItem);
            });
        } else {
            sourceDocumentsDiv.style.display = 'none';
        }

        queryAnswer.textContent = 'Query successful!';
        queryAnswer.classList.remove('alert-info');
        queryAnswer.classList.add('alert-success');
    } catch (error) {
        queryAnswer.textContent = `Error: ${error.message}`;
        queryAnswer.classList.remove('alert-info');
        queryAnswer.classList.add('alert-danger');
    }
});
