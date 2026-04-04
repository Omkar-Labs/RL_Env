// Setup basic state elements
const startBtn = document.getElementById('startTrainBtn');
const timestepsInput = document.getElementById('timesteps');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const resultsPanel = document.getElementById('resultsPanel');

const API_BASE_URL = "http://127.0.0.1:8000";

startBtn.addEventListener('click', async () => {
    const timesteps = parseInt(timestepsInput.value, 10);

    // Simple validation
    if (isNaN(timesteps) || timesteps < 1000) {
        alert("Please enter a valid number of timesteps (minimum 1000).");
        return;
    }

    // Update UI Loading State
    startBtn.classList.add('loading');
    startBtn.disabled = true;
    resultsPanel.classList.add('hidden');
    statusBadge.classList.add('hidden');

    try {
        // Send POST Request to FastAPI Backend
        const response = await fetch(`${API_BASE_URL}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ total_timesteps: timesteps })
        });

        if (!response.ok) {
            throw new Error(`Server responded with Status: ${response.status}`);
        }

        const data = await response.json();
        const r = data.results;

        // Inject results into the DOM
        document.getElementById('statAvgReward').textContent = r.avg_reward;
        document.getElementById('statMastery').textContent = `${r.mastery_pct}%`;
        document.getElementById('statBestReward').textContent = r.best_reward;
        document.getElementById('statWorstReward').textContent = r.worst_reward;
        document.getElementById('statFrustrations').textContent = r.total_frustrations;

        // Show success badge and results panel
        statusText.textContent = data.message;
        statusBadge.classList.remove('hidden');
        resultsPanel.classList.remove('hidden');

    } catch (error) {
        console.error("Training initialization error:", error);

        // Show error UI
        statusText.textContent = "Failed to connect to AI server. Is Uvicorn running?";
        statusText.style.color = "#ef4444";
        document.querySelector('.indicator').style.background = "#ef4444";
        document.querySelector('.indicator').style.boxShadow = "0 0 10px #ef4444";
        statusBadge.style.background = "rgba(239, 68, 68, 0.1)";
        statusBadge.style.borderColor = "rgba(239, 68, 68, 0.2)";
        statusBadge.classList.remove('hidden');

    } finally {
        // Reset Button Loading State
        startBtn.classList.remove('loading');
        startBtn.disabled = false;
    }
});
