/**
 * Hand Analysis Modal Component
 * 
 * Displays automatic hand analysis after each completed hand.
 */

class HandAnalysisModal {
    constructor() {
        this.modal = null;
        this.overlay = null;
        this.initializeModal();
    }
    
    initializeModal() {
        // Create modal structure
        this.overlay = document.createElement('div');
        this.overlay.id = 'handAnalysisModalOverlay';
        this.overlay.className = 'hand-analysis-overlay';
        
        this.modal = document.createElement('div');
        this.modal.id = 'handAnalysisModal';
        this.modal.className = 'hand-analysis-modal';
        this.modal.setAttribute('role', 'dialog');
        this.modal.setAttribute('aria-label', 'Hand Analysis');
        this.modal.setAttribute('aria-modal', 'true');
        
        // Modal content
        const content = document.createElement('div');
        content.className = 'hand-analysis-content';
        
        // Close button
        const closeBtn = document.createElement('button');
        closeBtn.className = 'hand-analysis-close';
        closeBtn.innerHTML = '×';
        closeBtn.setAttribute('aria-label', 'Close hand analysis modal');
        closeBtn.addEventListener('click', () => this.hide());
        
        
        // Decision list
        const decisionsSection = document.createElement('div');
        decisionsSection.className = 'hand-analysis-decisions';
        decisionsSection.id = 'handAnalysisDecisions';
        decisionsSection.setAttribute('role', 'region');
        decisionsSection.setAttribute('aria-label', 'Decision breakdown');
        
        // Key insights (Phase 1 rule-based)
        const insightsSection = document.createElement('div');
        insightsSection.className = 'hand-analysis-insights';
        insightsSection.id = 'handAnalysisInsights';
        
        // LLM-powered insights (Phase 4)
        const llmInsightsSection = document.createElement('div');
        llmInsightsSection.className = 'hand-analysis-llm-insights';
        llmInsightsSection.id = 'handAnalysisLLMInsights';
        
        // Pattern recognition (Phase 4)
        const patternSection = document.createElement('div');
        patternSection.className = 'hand-analysis-patterns';
        patternSection.id = 'handAnalysisPatterns';
        
        // Learning points (Phase 1 rule-based)
        const learningSection = document.createElement('div');
        learningSection.className = 'hand-analysis-learning';
        learningSection.id = 'handAnalysisLearning';
        
        // LLM-powered learning points (Phase 4)
        const llmLearningSection = document.createElement('div');
        llmLearningSection.className = 'hand-analysis-llm-learning';
        llmLearningSection.id = 'handAnalysisLLMLearning';
        
        // Assemble modal
        content.appendChild(closeBtn);
        content.appendChild(decisionsSection);
        content.appendChild(insightsSection);
        content.appendChild(llmInsightsSection);
        content.appendChild(patternSection);
        content.appendChild(learningSection);
        content.appendChild(llmLearningSection);
        
        this.modal.appendChild(content);
        this.overlay.appendChild(this.modal);
        
        // Add to document
        document.body.appendChild(this.overlay);
        
        // Close on overlay click
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hide();
            }
        });
        
        // Close on ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible()) {
                this.hide();
            }
        });
        
        // Trap focus in modal when visible
        this.modal.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' && this.isVisible()) {
                const focusableElements = this.modal.querySelectorAll(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                );
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        });
    }
    
    show(analysisData) {
        if (!analysisData) {
            return;
        }
        
        // Store previous focus for restoration
        this.previousFocus = document.activeElement;
        
        // Store analysis ID for async LLM retrieval
        this.analysisId = analysisData.analysis_id;
        this.llmEnhanced = analysisData.llm_enhanced || false;

        // Render decisions (with enhanced explanations if available)
        this.renderDecisions(analysisData.decisions || []);
        
        // Render Phase 1 rule-based insights
        this.renderInsights(analysisData.key_insights || [], 'rule-based');
        
        // Render Phase 1 rule-based learning points
        this.renderLearningPoints(analysisData.learning_points || [], 'rule-based');
        
        // Show loading indicators for LLM content
        if (this.analysisId && !this.llmEnhanced) {
            this.showLLMLoading();
        }
        
        // Show modal
        this.overlay.classList.add('show');
        this.modal.classList.add('show');
        
        // Focus first focusable element in modal
        const firstFocusable = this.modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) {
            firstFocusable.focus();
        }
        
        // Start polling for async LLM content if analysis_id is present
        if (this.analysisId && !this.llmEnhanced) {
            this.startAsyncPolling();
        }
    }
    
    hide() {
        this.overlay.classList.remove('show');
        this.modal.classList.remove('show');
        
        // Stop polling if active
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        
        // Restore focus to previous element
        if (this.previousFocus && typeof this.previousFocus.focus === 'function') {
            this.previousFocus.focus();
        }
        
        // Clear stored data
        this.analysisId = null;
        this.llmEnhanced = false;
        this.previousFocus = null;
        window.currentAnalysisData = null;
    }
    
    isVisible() {
        return this.overlay.classList.contains('show');
    }
    
    
    renderDecisions(decisions, enhancedExplanations = {}) {
        const decisionsSection = document.getElementById('handAnalysisDecisions');
        if (!decisionsSection) return;
        
        if (decisions.length === 0) {
            decisionsSection.innerHTML = '<p>No decisions to display</p>';
            return;
        }
        
        let html = '<h3>Decision Breakdown</h3><div class="decisions-list">';
        
        decisions.forEach((decision, index) => {
            const grade = decision.grade || 'C';
            const stage = decision.stage || 'unknown';
            const action = decision.action || 'Unknown';
            const explanation = decision.explanation || '';
            const enhancedExplanation = enhancedExplanations[index] || null;
            const optimal = decision.optimal_action || 'Unknown';
            const context = decision.context || {};
            
            const gradeDescriptions = {
                'A': 'Excellent',
                'B': 'Good',
                'C': 'Average',
                'D': 'Below Average',
                'F': 'Poor'
            };
            
            html += `
                <div class="decision-item" role="article" aria-label="Decision at ${stage}: ${action}, Grade ${grade}">
                    <div class="decision-header">
                        <span class="decision-stage" aria-label="Stage: ${stage}">${stage.toUpperCase()}</span>
                        <span class="decision-action" aria-label="Action: ${action}">${action}</span>
                        <span class="decision-grade grade-${grade}" aria-label="Grade ${grade}: ${gradeDescriptions[grade] || 'Unknown'}, ${decision.grade_percentage || 0}%">${grade} (${decision.grade_percentage || 0}%)</span>
                    </div>
                    <div class="decision-explanation">
                        <div class="explanation-rule-based">
                            <strong>Rule-Based:</strong> ${explanation}
                        </div>
                        ${enhancedExplanation ? `
                            <div class="explanation-enhanced">
                                <strong>AI-Enhanced:</strong> ${enhancedExplanation}
                            </div>
                        ` : ''}
                    </div>
                    <div class="decision-optimal">Optimal: ${optimal}</div>
                    ${this.renderDecisionContext(context)}
                </div>
            `;
        });
        
        html += '</div>';
        decisionsSection.innerHTML = html;
    }
    
    renderDecisionContext(context) {
        if (!context || Object.keys(context).length === 0) {
            return '';
        }
        
        let html = '<div class="decision-context">';
        
        if (context.hand) {
            html += `<span>Hand: ${context.hand}</span>`;
        }
        if (context.position) {
            html += `<span>Position: ${context.position}</span>`;
        }
        if (context.hand_strength) {
            html += `<span>Strength: ${context.hand_strength}</span>`;
        }
        if (context.pot_odds) {
            html += `<span>Pot Odds: ${context.pot_odds.toFixed(1)}:1</span>`;
        }
        if (context.equity) {
            html += `<span>Equity: ${context.equity}%</span>`;
        }
        
        html += '</div>';
        return html;
    }
    
    renderInsights(insights, type = 'rule-based') {
        const sectionId = type === 'rule-based' ? 'handAnalysisInsights' : 'handAnalysisLLMInsights';
        const insightsSection = document.getElementById(sectionId);
        if (!insightsSection) return;
        
        if (insights.length === 0) {
            insightsSection.innerHTML = '';
            return;
        }
        
        const label = type === 'rule-based' ? 'Key Insights (Rule-Based)' : 'Key Insights (AI-Enhanced)';
        let html = `<h3>${label}</h3><ul class="insights-list">`;
        insights.forEach(insight => {
            html += `<li>${insight}</li>`;
        });
        html += '</ul>';
        
        insightsSection.innerHTML = html;
    }
    
    renderPatterns(patternInsights) {
        const patternSection = document.getElementById('handAnalysisPatterns');
        if (!patternSection) return;
        
        if (!patternInsights || patternInsights.length === 0) {
            patternSection.innerHTML = '';
            return;
        }
        
        let html = '<h3>Pattern Recognition</h3><ul class="pattern-list">';
        patternInsights.forEach(pattern => {
            html += `<li class="pattern-item">${pattern}</li>`;
        });
        html += '</ul>';
        
        patternSection.innerHTML = html;
    }
    
    renderLearningPoints(learningPoints, type = 'rule-based') {
        const sectionId = type === 'rule-based' ? 'handAnalysisLearning' : 'handAnalysisLLMLearning';
        const learningSection = document.getElementById(sectionId);
        if (!learningSection) return;
        
        if (learningPoints.length === 0) {
            learningSection.innerHTML = '';
            return;
        }
        
        const label = type === 'rule-based' ? 'Learning Points (Rule-Based)' : 'Learning Points (AI-Enhanced)';
        let html = `<h3>${label}</h3><ul class="learning-list">`;
        learningPoints.forEach(point => {
            html += `<li>${point}</li>`;
        });
        html += '</ul>';
        
        learningSection.innerHTML = html;
    }
    
    showLLMLoading() {
        // Show loading indicators
        const llmInsightsSection = document.getElementById('handAnalysisLLMInsights');
        if (llmInsightsSection) {
            llmInsightsSection.innerHTML = '<h3>Key Insights (AI-Enhanced)</h3><div class="loading-indicator">Loading AI insights...</div>';
        }
        
        const patternSection = document.getElementById('handAnalysisPatterns');
        if (patternSection) {
            patternSection.innerHTML = '<h3>Pattern Recognition</h3><div class="loading-indicator">Analyzing patterns...</div>';
        }
        
        const llmLearningSection = document.getElementById('handAnalysisLLMLearning');
        if (llmLearningSection) {
            llmLearningSection.innerHTML = '<h3>Learning Points (AI-Enhanced)</h3><div class="loading-indicator">Generating learning points...</div>';
        }
    }
    
    startAsyncPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        
        let pollCount = 0;
        const maxPolls = 10; // 5 seconds max (500ms * 10)
        
        this.pollInterval = setInterval(async () => {
            pollCount++;
            
            if (pollCount > maxPolls) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                // Show timeout message
                this.showLLMTimeout();
                return;
            }
            
            try {
                const response = await fetch(`/api/coach/analyze-hand-async/${this.analysisId}`);
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.ready) {
                        // LLM content is ready, update display
                        clearInterval(this.pollInterval);
                        this.pollInterval = null;
                        this.updateWithLLMContent(data);
                    }
                } else {
                    // If we get a non-200 response, log it but continue polling
                }
            } catch (error) {
                // Continue polling on error (network issues might be temporary)
            }
        }, 500); // Poll every 500ms
    }
    
    updateWithLLMContent(llmData) {
        // Check if API is unavailable
        if (llmData.api_unavailable) {
            this.showAPINotAvailable();
            return;
        }
        
        // Check for errors
        if (llmData.error) {
            this.showLLMError(llmData.error);
            return;
        }
        
        // Update LLM insights
        if (llmData.llm_insights && llmData.llm_insights.length > 0) {
            this.renderInsights(llmData.llm_insights, 'llm');
        } else {
            // No insights generated - show message
            const llmInsightsSection = document.getElementById('handAnalysisLLMInsights');
            if (llmInsightsSection) {
                llmInsightsSection.innerHTML = '<h3>Key Insights (AI-Enhanced)</h3><div class="info-message">No AI insights generated - showing rule-based insights only</div>';
            }
        }
        
        // Update pattern recognition
        if (llmData.pattern_insights && llmData.pattern_insights.length > 0) {
            this.renderPatterns(llmData.pattern_insights);
        }
        
        // Update enhanced explanations
        if (llmData.enhanced_explanations && Object.keys(llmData.enhanced_explanations).length > 0) {
            const decisionsSection = document.getElementById('handAnalysisDecisions');
            if (decisionsSection) {
                // Re-render decisions with enhanced explanations
                const decisions = window.currentAnalysisData?.decisions || [];
                this.renderDecisions(decisions, llmData.enhanced_explanations);
            }
        }
        
        // Update LLM learning points
        if (llmData.llm_learning_points && llmData.llm_learning_points.length > 0) {
            this.renderLearningPoints(llmData.llm_learning_points, 'llm');
        }
    }
    
    showAPINotAvailable() {
        // Show API not available message
        const llmInsightsSection = document.getElementById('handAnalysisLLMInsights');
        if (llmInsightsSection) {
            llmInsightsSection.innerHTML = '<h3>Key Insights (AI-Enhanced)</h3><div class="info-message">AI insights unavailable - API key not configured. Please set OPEN_ROUTER_KEY or OPENAI_API_KEY environment variable.</div>';
        }
        
        const patternSection = document.getElementById('handAnalysisPatterns');
        if (patternSection) {
            patternSection.innerHTML = '';
        }
        
        const llmLearningSection = document.getElementById('handAnalysisLLMLearning');
        if (llmLearningSection) {
            llmLearningSection.innerHTML = '';
        }
    }
    
    showLLMError(errorMessage) {
        // Show error message
        const llmInsightsSection = document.getElementById('handAnalysisLLMInsights');
        if (llmInsightsSection) {
            llmInsightsSection.innerHTML = `<h3>Key Insights (AI-Enhanced)</h3><div class="error-message">AI insights unavailable - error: ${errorMessage}. Showing rule-based insights only.</div>`;
        }
        
        const patternSection = document.getElementById('handAnalysisPatterns');
        if (patternSection) {
            patternSection.innerHTML = '';
        }
        
        const llmLearningSection = document.getElementById('handAnalysisLLMLearning');
        if (llmLearningSection) {
            llmLearningSection.innerHTML = '';
        }
    }
    
    showLLMTimeout() {
        // Show timeout message
        const llmInsightsSection = document.getElementById('handAnalysisLLMInsights');
        if (llmInsightsSection) {
            llmInsightsSection.innerHTML = '<h3>Key Insights (AI-Enhanced)</h3><div class="timeout-message">AI insights unavailable - showing rule-based insights only</div>';
        }
        
        const patternSection = document.getElementById('handAnalysisPatterns');
        if (patternSection) {
            patternSection.innerHTML = '';
        }
        
        const llmLearningSection = document.getElementById('handAnalysisLLMLearning');
        if (llmLearningSection) {
            llmLearningSection.innerHTML = '';
        }
    }
}

// Initialize modal when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.handAnalysisModal = new HandAnalysisModal();
    });
} else {
    window.handAnalysisModal = new HandAnalysisModal();
}

/**
 * Chat Manager Component
 * 
 * Manages the AI coach chat interface.
 */
class ChatManager {
    constructor() {
        this.chatHistory = document.getElementById('chatHistory');
        this.chatInput = document.getElementById('chatInput');
        this.chatSendBtn = document.getElementById('chatSendBtn');
        this.typingIndicator = document.getElementById('chatTypingIndicator');
        this.isSending = false;
        this.sessionId = null;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Send button click
        if (this.chatSendBtn) {
            this.chatSendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        // Enter key to send
        if (this.chatInput) {
            this.chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
    }
    
    setSessionId(sessionId) {
        this.sessionId = sessionId;
    }
    
    async sendMessage() {
        // Prevent rapid sending
        if (this.isSending) {
            return;
        }
        
        // Get message
        const message = this.chatInput.value.trim();
        
        // Validate message
        if (!message) {
            return;
        }
        
        // Get session ID
        const sessionId = this.sessionId || (window.pokerGame && window.pokerGame.sessionId) || 'default';
        
        // Get game context if available
        const gameContext = this.getGameContext();
        
        // Disable input and send button
        this.isSending = true;
        this.chatInput.disabled = true;
        this.chatSendBtn.disabled = true;
        
        // Display user message
        this.addMessage('user', message);
        
        // Clear input
        this.chatInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Call API
            const response = await fetch('/api/coach/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message,
                    game_context: gameContext
                })
            });
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            if (response.ok) {
                // Display coach response
                this.addMessage('coach', data.response || data.error || 'No response received');
            } else {
                // Error response - user-friendly messages
                let errorMsg = 'Coach unavailable, please try again';
                if (data.error) {
                    if (data.error.includes('timeout') || data.error.includes('time')) {
                        errorMsg = 'Coach is taking longer than expected. Please try again in a moment.';
                    } else if (data.error.includes('network') || data.error.includes('connection')) {
                        errorMsg = 'Connection issue. Please check your network and try again.';
                    } else if (data.error.includes('rate limit') || data.error.includes('rate')) {
                        errorMsg = 'Too many requests. Please wait a moment and try again.';
                    } else {
                        errorMsg = data.error;
                    }
                }
                this.addMessage('coach', errorMsg, true);
            }
            
        } catch (error) {
            // Network error - user-friendly message
            let errorMsg = 'Unable to connect to coach. Please check your connection and try again.';
            if (error.message && error.message.includes('timeout')) {
                errorMsg = 'Request timed out. Please try again.';
            } else if (error.message && error.message.includes('Failed to fetch')) {
                errorMsg = 'Connection failed. Please check your network.';
            } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
                errorMsg = 'Network error. Please check your connection and try again.';
            }
            this.hideTypingIndicator();
            
            this.addMessage('coach', errorMsg, true);
        } finally {
            // Re-enable input and send button
            this.isSending = false;
            this.chatInput.disabled = false;
            this.chatSendBtn.disabled = false;
            this.chatInput.focus();
        }
    }
    
    getGameContext() {
        // Get game context from PokerGame if available
        if (window.pokerGame && window.pokerGame.gameState) {
            return window.pokerGame.gameState;
        }
        return null;
    }
    
    addMessage(role, content, isError = false) {
        if (!this.chatHistory) {
            return;
        }
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        if (isError) {
            messageDiv.style.borderLeft = '3px solid #f44336';
        }
        
        // Message content - render markdown for coach messages
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        if (role === 'coach' && !isError) {
            // Render markdown for coach responses
            messageContent.innerHTML = this.renderMarkdown(content);
        } else {
            // Plain text for user messages and errors
            messageContent.textContent = content;
        }
        messageDiv.appendChild(messageContent);
        
        // Timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-time';
        const now = new Date();
        timestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageDiv.appendChild(timestamp);
        
        // Add to history
        this.chatHistory.appendChild(messageDiv);
        
        // Auto-scroll to bottom
        this.scrollToBottom();
    }
    
    renderMarkdown(text) {
        if (!text) return '';
        
        // Escape HTML to prevent XSS
        const escapeHtml = (str) => {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        };
        
        // Split into lines for processing
        const lines = text.split('\n');
        const result = [];
        let inList = false;
        let listType = null; // 'ul' or 'ol'
        let listItems = [];
        
        const flushList = () => {
            if (listItems.length > 0) {
                const tag = listType === 'ol' ? 'ol' : 'ul';
                result.push(`<${tag}>${listItems.join('')}</${tag}>`);
                listItems = [];
            }
            inList = false;
            listType = null;
        };
        
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();
            
            // Check for markdown headers (#, ##, ###, etc.)
            const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
            if (headerMatch) {
                // Flush any active list before header
                if (inList) {
                    flushList();
                }
                const level = headerMatch[1].length; // Number of # characters
                const headerText = headerMatch[2];
                const processed = this.processInlineMarkdown(headerText);
                result.push(`<h${level}>${processed}</h${level}>`);
                continue;
            }
            
            // Check for numbered list (1., 2., etc.)
            const numberedMatch = line.match(/^(\d+)\.\s+(.+)$/);
            if (numberedMatch) {
                if (!inList || listType !== 'ol') {
                    flushList();
                    inList = true;
                    listType = 'ol';
                }
                const content = this.processInlineMarkdown(numberedMatch[2]);
                listItems.push(`<li>${content}</li>`);
                continue;
            }
            
            // Check for bullet list (- or *)
            const bulletMatch = line.match(/^[-*]\s+(.+)$/);
            if (bulletMatch) {
                if (!inList || listType !== 'ul') {
                    flushList();
                    inList = true;
                    listType = 'ul';
                }
                const content = this.processInlineMarkdown(bulletMatch[1]);
                listItems.push(`<li>${content}</li>`);
                continue;
            }
            
            // Not a list item - flush any active list
            if (inList) {
                flushList();
            }
            
            // Process regular line
            if (line) {
                const processed = this.processInlineMarkdown(line);
                result.push(`<p>${processed}</p>`);
            } else {
                // Empty line - add spacing
                result.push('<br>');
            }
        }
        
        // Flush any remaining list
        flushList();
        
        return result.join('');
    }
    
    processInlineMarkdown(text) {
        // Escape HTML first
        const escapeHtml = (str) => {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        };
        
        let html = escapeHtml(text);
        
        // Bold: **text** or __text__
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
        
        // Italic: *text* or _text_ (but not if part of bold or surrounded by word chars)
        // Use word boundaries to avoid matching within words
        html = html.replace(/(^|[^*])\*([^*]+)\*([^*]|$)/g, '$1<em>$2</em>$3');
        html = html.replace(/(^|[^_])_([^_]+)_([^_]|$)/g, '$1<em>$2</em>$3');
        
        return html;
    }
    
    showTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'block';
            this.scrollToBottom();
        }
    }
    
    hideTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'none';
        }
    }
    
    scrollToBottom() {
        if (this.chatHistory) {
            this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
        }
    }
    
    clearHistory() {
        if (this.chatHistory) {
            this.chatHistory.innerHTML = '';
        }
    }
}

// Initialize chat manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.chatManager = new ChatManager();
    });
} else {
    window.chatManager = new ChatManager();
}

