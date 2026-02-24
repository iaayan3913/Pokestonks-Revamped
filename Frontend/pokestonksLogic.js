        // TCGdex — free, open-source Pokémon TCG API (no key required)
        const TCGDEX_BASE = "https://api.tcgdex.net/v2/en";
        const BACKEND_URL = "http://127.0.0.1:8000";

        // create a public chart variable
        let chart = null;

        // Currently viewed card data (for AI context)
        let activeCardData = null;
        let lastForecastData = null;

        // Pagination variables
        let currentCards = [];
        let currentPage = 1;
        const cardsPerPage = 15;

        // Add caching to prevent repeated API calls
        const cardCache = new Map();
        const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

        const searchForm = document.getElementById('searchForm');  // use getElementById to access  these divs
        const loadingDiv = document.getElementById('loading');
        const resultsInfo = document.getElementById('resultsInfo');
        const cardsContainer = document.getElementById('cardsContainer');
        const noResults = document.getElementById('noResults');

        searchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const cardName = document.getElementById('cardName').value.trim(); //store the value  of the card name and card ID fields
            const cardId = document.getElementById('cardId').value.trim();
            
            if (!cardName && !cardId) {
                alert('Please enter either a Pokemon name or card ID');
                return;
            }
            
            await searchCards(cardName, cardId);
        });




        function getCacheKey(cardName, cardId) {
            return cardId || cardName.toLowerCase();  //use id of name. make name all lowercase to make searching easier
        }

        function getCachedData(key) {
            const cached = cardCache.get(key);
            if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
                return cached.data;
            }
            return null;
        }

        function setCachedData(key, data) {
            cardCache.set(key, {
                data: data,
                timestamp: Date.now()
            });
        }

        async function searchCards(cardName, cardId) {
            const cacheKey = getCacheKey(cardName || '', cardId || '');
            
            // Check cache first
            const cachedCards = getCachedData(cacheKey);
            if (cachedCards) {
                displayCards(cachedCards);
                return;
            }
            
            showLoading();
            
            try {
                let cards;
                if (cardId) {
                    cards = await fetchCardById(cardId);
                } else {
                    cards = await fetchCardsByName(cardName);
                }
                
                // Cache the results
                setCachedData(cacheKey, cards);
                displayCards(cards);
            } catch (error) {
                console.error('Error fetching cards:', error);
                showError(error.message);
            }
        }

        // ---- TCGdex data normaliser ----
        // Maps TCGdex card shape → the format the rest of the UI expects
        function normaliseTcgdexCard(raw) {
            // Extract pricing into the tcgplayer-style object the chart panel uses
            const pricing = raw.pricing || {};
            const tcgp = pricing.tcgplayer || {};
            const cm   = pricing.cardmarket || {};

            // Build a "prices" object keyed by variant name
            const prices = {};
            for (const [variant, data] of Object.entries(tcgp)) {
                if (variant === 'updated' || variant === 'unit') continue;
                prices[variant] = {
                    low:    data.lowPrice    ?? null,
                    mid:    data.midPrice    ?? null,
                    high:   data.highPrice   ?? null,
                    market: data.marketPrice ?? null,
                };
            }

            // If no TCGplayer data, try Cardmarket
            if (Object.keys(prices).length === 0 && cm.avg) {
                prices['cardmarket'] = {
                    low:    cm.low   ?? null,
                    mid:    cm.avg   ?? null,
                    high:   cm.trend ?? null,
                    market: cm.avg30 ?? null,
                };
            }

            return {
                id:     raw.id,
                name:   raw.name,
                rarity: raw.rarity || 'Unknown',
                hp:     raw.hp     || null,
                types:  raw.types  || [],
                stage:  raw.stage  || null,
                images: {
                    small: raw.image ? `${raw.image}/low.webp`  : null,
                    large: raw.image ? `${raw.image}/high.webp` : null,
                },
                set: {
                    name:        raw.set?.name       || 'Unknown',
                    id:          raw.set?.id          || '',
                    series:      raw.set?.serie       || raw.set?.series || 'Unknown',
                    releaseDate: raw.set?.releaseDate || 'Unknown',
                    logo:        raw.set?.logo ? `${raw.set.logo}.webp` : null,
                },
                tcgplayer: { prices },
                _raw: raw,   // keep original for AI context
            };
        }

        async function fetchCardsByName(cardName) {
            const maxResults = 100;

            // TCGdex search: /v2/en/cards?name=<name>
            const url = `${TCGDEX_BASE}/cards?name=${encodeURIComponent(cardName)}`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`TCGdex error: ${response.status}`);
            }

            const list = await response.json();        // array of {id, localId, name, image}
            if (!Array.isArray(list) || list.length === 0) return [];

            // The list endpoint only returns id/name/image.
            // Fetch full details (with pricing) for up to maxResults cards.
            const toFetch = list.slice(0, maxResults);

            // Batch fetch in groups of 10 to be polite to the API
            const fullCards = [];
            for (let i = 0; i < toFetch.length; i += 10) {
                const batch = toFetch.slice(i, i + 10);
                const results = await Promise.all(
                    batch.map(c =>
                        fetch(`${TCGDEX_BASE}/cards/${c.id}`)
                            .then(r => r.ok ? r.json() : null)
                            .catch(() => null)
                    )
                );
                fullCards.push(...results.filter(Boolean));
            }

            return removeDuplicatesById(fullCards.map(normaliseTcgdexCard));
        }

        async function fetchCardById(cardId) {
            const response = await fetch(`${TCGDEX_BASE}/cards/${cardId}`);

            if (!response.ok) {
                throw new Error(`TCGdex error: ${response.status}`);
            }

            const card = await response.json();
            return [normaliseTcgdexCard(card)];
        }

        function removeDuplicatesById(cards) {
            const seenIds = new Set();
            return cards.filter(card => {
                if (seenIds.has(card.id)) {
                    return false;
                }
                seenIds.add(card.id);
                return true;
            });
        }

        function displayCards(cards) {
            hideLoading();
            
            if (cards.length === 0) {
                showNoResults();
                return;
            }

            currentCards = cards;
            currentPage = 1;
            
            showResults(cards.length);
            renderCurrentPage();
            renderPagination();
        }

        function renderCurrentPage() {
            const startIndex = (currentPage - 1) * cardsPerPage;
            const endIndex = startIndex + cardsPerPage;
            const cardsToShow = currentCards.slice(startIndex, endIndex);
            
            cardsContainer.innerHTML = cardsToShow.map(card => createCardHTML(card)).join('');
            
            // Scroll to top of results
            document.getElementById('resultsInfo').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }

        function renderPagination() {
            const totalPages = Math.ceil(currentCards.length / cardsPerPage);
            
            if (totalPages <= 1) {
                document.getElementById('paginationTop').style.display = 'none';
                document.getElementById('paginationBottom').style.display = 'none';
                return;
            }

            const paginationHTML = createPaginationHTML(totalPages);
            document.getElementById('paginationTop').innerHTML = paginationHTML;
            document.getElementById('paginationBottom').innerHTML = paginationHTML;
            document.getElementById('paginationTop').style.display = 'flex';
            document.getElementById('paginationBottom').style.display = 'flex';

            // Add event listeners to pagination buttons
            addPaginationListeners();
        }

        function createPaginationHTML(totalPages) {
            let html = '';
            
            // Previous button
            html += `<button class="pagination-btn" ${currentPage === 1 ? 'disabled' : ''} data-page="prev">
                        ‹ Previous
                     </button>`;
            
            // Page numbers
            html += '<div class="page-numbers">';
            
            // Show first page
            if (currentPage > 3) {
                html += `<button class="pagination-btn" data-page="1">1</button>`;
                if (currentPage > 4) {
                    html += `<span class="pagination-info">...</span>`;
                }
            }
            
            // Show pages around current page
            const startPage = Math.max(1, currentPage - 2);
            const endPage = Math.min(totalPages, currentPage + 2);
            
            for (let i = startPage; i <= endPage; i++) {
                html += `<button class="pagination-btn ${i === currentPage ? 'active' : ''}" data-page="${i}">
                            ${i}
                         </button>`;
            }
            
            // Show last page
            if (currentPage < totalPages - 2) {
                if (currentPage < totalPages - 3) {
                    html += `<span class="pagination-info">...</span>`;
                }
                html += `<button class="pagination-btn" data-page="${totalPages}">${totalPages}</button>`;
            }
            
            html += '</div>';
            
            // Next button
            html += `<button class="pagination-btn" ${currentPage === totalPages ? 'disabled' : ''} data-page="next">
                        Next ›
                     </button>`;
            
            // Page info
            const startIndex = (currentPage - 1) * cardsPerPage + 1;
            const endIndex = Math.min(currentPage * cardsPerPage, currentCards.length);
            html += `<div class="pagination-info">
                        Showing ${startIndex}-${endIndex} of ${currentCards.length} cards
                     </div>`;
            
            return html;
        }

        function addPaginationListeners() {
            const paginationBtns = document.querySelectorAll('.pagination-btn');
            paginationBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const page = e.target.getAttribute('data-page');
                    
                    if (page === 'prev' && currentPage > 1) {
                        currentPage--;
                    } else if (page === 'next' && currentPage < Math.ceil(currentCards.length / cardsPerPage)) {
                        currentPage++;
                    } else if (page !== 'prev' && page !== 'next') {
                        currentPage = parseInt(page);
                    }
                    
                    renderCurrentPage();
                    renderPagination();
                });
            });
        }

        function createCardHTML(card) {
            const rarity = card.rarity || 'Unknown';
            const rarityClass = `rarity-${rarity.toLowerCase().replace(/\s+/g, '-')}`;
            
            return `
                <div class="card">
                    <div class="card-image">
                        <img src="${card.images?.small || 'https://via.placeholder.com/150x210?text=No+Image'}" 
                             alt="${card.name}" 
                             loading="lazy"
                             onclick="showChart('${card.id}', '${card.name.replace(/'/g, "\\'")}', '${card.images?.large || card.images?.small || ''}')"
                             class="clickable-card-image">
                    </div>
                    <div class="card-details">
                        <h3>${card.name}</h3>
                        <div class="card-info">
                            <div class="info-row">
                                <span class="info-label">Card ID:</span>
                                <span class="info-value">${card.id}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Set:</span>
                                <span class="info-value">${card.set?.name || 'Unknown'}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Rarity:</span>
                                <span class="info-value">
                                    <span class="rarity-badge ${rarityClass}">${rarity}</span>
                                </span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Release Date:</span>
                                <span class="info-value">${card.set?.releaseDate || 'Unknown'}</span>
            </div>
                            <div class="info-row">
                                <span class="info-label">Series:</span>
                                <span class="info-value">${card.set?.series || 'Unknown'}</span>
                            </div>
                            ${card.hp ? `
                            <div class="info-row">
                                <span class="info-label">HP:</span>
                                <span class="info-value">${card.hp}</span>
                            </div>
                            ` : ''}
                            ${card.types ? `
                            <div class="info-row">
                                <span class="info-label">Type:</span>
                                <span class="info-value">${card.types.join(', ')}</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        }

        function showLoading() {
            loadingDiv.style.display = 'block';
            resultsInfo.style.display = 'none';
            cardsContainer.innerHTML = '';
            noResults.style.display = 'none';
            hideAllPagination();
        }

        function hideLoading() {
            loadingDiv.style.display = 'none';
        }

        function showResults(count) {
            resultsInfo.textContent = `Found ${count} unique card${count !== 1 ? 's' : ''}`;
            resultsInfo.style.display = 'block';
            noResults.style.display = 'none';
        }

        function hideAllPagination() {
            document.getElementById('paginationTop').style.display = 'none';
            document.getElementById('paginationBottom').style.display = 'none';
        }

        function showError(message) {
            hideLoading();
            hideAllPagination();
            resultsInfo.style.display = 'none';
            cardsContainer.innerHTML = '';
            noResults.innerHTML = `
                <h3>Error occurred</h3>
                <p>${message}</p>
                <p>Please try again or check your API key if you're hitting rate limits.</p>
            `;
            noResults.style.display = 'block';
        }

        function showNoResults() {
            hideLoading();
            hideAllPagination();
            resultsInfo.style.display = 'none';
            cardsContainer.innerHTML = '';
            noResults.innerHTML = `
                <h3>No cards found</h3>
                <p>Try searching with a different Pokemon name or card ID</p>
            `;
            noResults.style.display = 'block';
        }

        document.addEventListener('DOMContentLoaded', () => {
            checkAIStatus();
        });

async function LoadForecast(cardId) { 
    try {
    const response = await fetch(`${BACKEND_URL}/forecast/${cardId}`);
    const data = await response.json();

    // Store forecast data for AI analysis
    lastForecastData = data;

    const labels = data.historical.map(p => p.date)
                .concat(data.forecast.map(f => f.date));
    const histPrices = data.historical.map(p => p.price);
    const forecastPrices = data.forecast.map(f => f.price);
    const lower = data.forecast.map(f => f.lower);
    const upper = data.forecast.map(f => f.upper);

    console.log("labels:", labels);
    console.log("historical prices:", histPrices);
    console.log("forecast prices:", forecastPrices);
    console.log("lower bounds:", lower);
    console.log("upper bounds:", upper);

    chart = new Chart(document.getElementById("forecastChart"), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
            {
                label: 'Historical',
                data: histPrices,
                borderColor: 'blue',
                pointRadius: 2,
                fill: false
            },
            {
                label: 'Forecast',
                data: [...Array(histPrices.length).fill(null), ...forecastPrices],
                borderColor: 'red',
                borderDash: [5, 5],
                pointRadius: 2,
                fill: false
            },
            {
                label: 'Confidence Range',
                data: [...Array(histPrices.length).fill(null), ...upper],
                borderColor: 'transparent',
                backgroundColor: 'rgba(255,0,0,0.1)',
                fill: '-1'
            },
            {
                label: '',
                data: [...Array(histPrices.length).fill(null), ...lower],
                borderColor: 'transparent',
                backgroundColor: 'rgba(255,0,0,0.1)',
                fill: false
            }
            ]
        },
        options: {
            responsive: true,
            interaction: {
            mode: 'index',
            intersect: false
            },
            plugins: {
            tooltip: {
                enabled: true
            },
            legend: {
                position: 'top'
            }
            },
            scales: {
            x: { title: { display: true, text: 'Date' }},
            y: { title: { display: true, text: 'Price ($)' }}
            }
        }
    });

    // Chart rendered — hide the loading overlay
    setForecastStatus('hidden');

    // Automatically trigger AI forecast analysis
    getAIForecastAnalysis();
    } catch (err) {
        console.error("Failed to load forecast:", err);
        setForecastStatus('nodata');
    }
}

function setForecastStatus(state) {
    // state: 'loading' | 'nodata' | 'hidden'
    const overlay   = document.getElementById('forecastStatus');
    const loading   = document.getElementById('forecastLoading');
    const noData    = document.getElementById('forecastNoData');
    const canvas    = document.getElementById('forecastChart');

    if (state === 'loading') {
        overlay.style.display = 'flex';
        loading.style.display = 'flex';
        noData.style.display  = 'none';
        canvas.style.display  = 'none';
    } else if (state === 'nodata') {
        overlay.style.display = 'flex';
        loading.style.display = 'none';
        noData.style.display  = 'flex';
        canvas.style.display  = 'none';
    } else {
        // hidden — chart is ready
        overlay.style.display = 'none';
        loading.style.display = 'none';
        noData.style.display  = 'none';
        canvas.style.display  = 'block';
    }
}

async function checkExistence(cardId) {
    // Start by showing loading while we check
    setForecastStatus('loading');

    try {
        const res = await fetch(`${BACKEND_URL}/has_data/${cardId}`);
        const json = await res.json();

        if (json.exists) {
            console.log("Forecasting data found, loading...");
            LoadForecast(cardId);
        } else {
            console.log("No forecast data for this card");
            setForecastStatus('nodata');
        }
    } catch (err) {
        console.log("Backend not reachable for forecast check:", err.message);
        setForecastStatus('nodata');
    }
}

function showChart(cardID, cardName, cardIMG) {
    // Show the chart container
    const container = document.getElementById('chart-container');
    container.style.display = 'block';
    
    // Update title 
    document.getElementById("chart").innerHTML = `${cardID} ⟹ ${cardName}`;
    
    // Update image src (use large image for chart view)
    const imgElement = document.getElementById("chartImage");
    imgElement.src = cardIMG;
    
    // Find the full card object from currentCards
    const fullCard = currentCards.find(c => c.id === cardID);
    const tcgplayerData = fullCard?.tcgplayer || {};

    activeCardData = {
        id: cardID,
        name: cardName,
        image: cardIMG,
        set: fullCard?.set?.name || 'Unknown',
        rarity: fullCard?.rarity || 'Unknown',
        types: fullCard?.types?.join(', ') || 'Unknown',
        hp: fullCard?.hp || 'N/A',
        prices: null,
        tcgplayer: tcgplayerData
    };

    if (!tcgplayerData.prices || Object.keys(tcgplayerData.prices).length === 0) {
        document.getElementById("prices").innerHTML = `No pricing data available`;
    } else { 
        // Try multiple price categories
        const priceCategory = tcgplayerData.prices.normal || tcgplayerData.prices.holofoil || 
                              tcgplayerData.prices.reverseHolofoil || tcgplayerData.prices['1stEditionHolofoil'] ||
                              tcgplayerData.prices.cardmarket ||
                              Object.values(tcgplayerData.prices)[0];
        
        if (priceCategory) {
            const low = priceCategory.low || null;
            const mid = priceCategory.mid || null;
            const high = priceCategory.high || null;
            const market = priceCategory.market || null;

            // Detect currency (Cardmarket = EUR, TCGplayer = USD)
            const isCardmarket = !!(tcgplayerData.prices.cardmarket && !tcgplayerData.prices.normal && !tcgplayerData.prices.holofoil);
            const symbol = isCardmarket ? '€' : '$';
            
            activeCardData.prices = { low, mid, high, market };
            
            let priceHTML = '';
            if (low) priceHTML += `Lowest: ${symbol}${low}<br>`;
            if (mid) priceHTML += `Average: ${symbol}${mid}<br>`;
            if (high) priceHTML += `Highest: ${symbol}${high}<br>`;
            if (market) priceHTML += `Market: ${symbol}${market}`;
            
            document.getElementById("prices").innerHTML = priceHTML || 'No pricing data available';
            console.log("prices updated");
        } else { 
            document.getElementById("prices").innerHTML = `No pricing data available`;
        }
    }

    // Reset AI panels
    document.getElementById('aiAnalysisPanel').style.display = 'none';
    document.getElementById('aiRecommendationsPanel').style.display = 'none';
    lastForecastData = null;

    checkExistence(cardID);
}

function closeChart() {
    if (chart) {
        chart.destroy();
        chart = null;
        console.log("chart destroyed");
    } 

    const container = document.getElementById('chart-container');
    container.style.display = 'none';
    
    // Reset AI state
    activeCardData = null;
    lastForecastData = null;
    document.getElementById('aiAnalysisPanel').style.display = 'none';
    document.getElementById('aiRecommendationsPanel').style.display = 'none';
    console.log("chart container hidden");
}

// ==================== AI FEATURES ====================

function getSelectedModel() {
    return document.getElementById('aiModel').value;
}

async function checkAIStatus() {
    const statusEl = document.getElementById('aiStatus');
    try {
        const res = await fetch(`${BACKEND_URL}/ai/status`);
        const data = await res.json();
        
        if (data.claude || data.gemini) {
            const models = [];
            if (data.claude) models.push('Claude');
            if (data.gemini) models.push('Gemini');
            statusEl.textContent = `✓ ${models.join(' & ')} ready`;
            statusEl.className = 'ai-status connected';
            
            // Auto-select first available model
            if (!data.claude && data.gemini) {
                document.getElementById('aiModel').value = 'gemini';
            }
        } else {
            statusEl.textContent = '✕ No AI keys configured';
            statusEl.className = 'ai-status disconnected';
        }
    } catch (err) {
        statusEl.textContent = '✕ Backend offline';
        statusEl.className = 'ai-status disconnected';
    }
}

// ==================== AI Chat ====================

function toggleAIChat() {
    const panel = document.getElementById('aiChatPanel');
    panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
}

async function sendAIChat() {
    const input = document.getElementById('aiChatInput');
    const message = input.value.trim();
    if (!message) return;

    const messagesDiv = document.getElementById('aiChatMessages');

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'ai-message user';
    userMsg.innerHTML = `<div class="ai-message-content">${escapeHTML(message)}</div>`;
    messagesDiv.appendChild(userMsg);

    input.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Add typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'ai-message bot';
    typingDiv.id = 'ai-typing';
    typingDiv.innerHTML = `<div class="ai-typing-indicator"><span></span><span></span><span></span></div>`;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        const res = await fetch(`${BACKEND_URL}/ai/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                model: getSelectedModel(),
                card_context: activeCardData ? {
                    id: activeCardData.id,
                    name: activeCardData.name,
                    set: activeCardData.set,
                    rarity: activeCardData.rarity,
                    prices: activeCardData.prices ? JSON.stringify(activeCardData.prices) : 'N/A'
                } : null
            })
        });

        const data = await res.json();
        
        // Remove typing indicator
        const typing = document.getElementById('ai-typing');
        if (typing) typing.remove();

        // Add bot response
        const botMsg = document.createElement('div');
        botMsg.className = 'ai-message bot';
        botMsg.innerHTML = `<div class="ai-message-content">${marked.parse(data.reply)}</div>`;
        messagesDiv.appendChild(botMsg);

    } catch (err) {
        const typing = document.getElementById('ai-typing');
        if (typing) typing.remove();

        const errMsg = document.createElement('div');
        errMsg.className = 'ai-message bot';
        errMsg.innerHTML = `<div class="ai-message-content">⚠️ Couldn't reach AI. Make sure the backend is running on ${BACKEND_URL}</div>`;
        messagesDiv.appendChild(errMsg);
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ==================== AI Forecast Analysis ====================

async function getAIForecastAnalysis() {
    if (!lastForecastData || !activeCardData) return;

    const panel = document.getElementById('aiAnalysisPanel');
    const content = document.getElementById('aiAnalysisContent');
    
    panel.style.display = 'block';
    content.innerHTML = '<div class="ai-typing-indicator"><span></span><span></span><span></span></div> Analyzing price forecast...';

    try {
        const res = await fetch(`${BACKEND_URL}/ai/forecast-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card_id: activeCardData.id,
                card_name: activeCardData.name,
                model: getSelectedModel(),
                historical: lastForecastData.historical,
                forecast: lastForecastData.forecast,
                current_prices: activeCardData.prices
            })
        });

        const data = await res.json();
        content.innerHTML = marked.parse(data.analysis);

    } catch (err) {
        content.innerHTML = `⚠️ Could not generate analysis. Backend may be offline.`;
    }
}

function closeAIAnalysis() {
    document.getElementById('aiAnalysisPanel').style.display = 'none';
}

// ==================== AI Card Recommendations ====================

async function getAIRecommendations() {
    if (!activeCardData) {
        alert('Select a card first to get recommendations.');
        return;
    }

    const panel = document.getElementById('aiRecommendationsPanel');
    const content = document.getElementById('aiRecommendationsContent');
    
    panel.style.display = 'block';
    content.innerHTML = '<div class="ai-typing-indicator"><span></span><span></span><span></span></div> Finding recommendations...';

    try {
        const res = await fetch(`${BACKEND_URL}/ai/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                card_name: activeCardData.name,
                card_type: activeCardData.types,
                card_rarity: activeCardData.rarity,
                card_set: activeCardData.set,
                current_price: activeCardData.prices?.mid || activeCardData.prices?.market || null,
                model: getSelectedModel()
            })
        });

        const data = await res.json();
        content.innerHTML = marked.parse(data.recommendations);

    } catch (err) {
        content.innerHTML = `⚠️ Could not get recommendations. Backend may be offline.`;
    }
}

function closeRecommendations() {
    document.getElementById('aiRecommendationsPanel').style.display = 'none';
}

// ==================== Utility ====================

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}