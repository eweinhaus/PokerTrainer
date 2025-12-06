/**
 * No Limit Texas Hold'em Web App - Frontend JavaScript
 */

class PokerGame {
    constructor() {
        this.sessionId = 'game_' + Date.now();
        this.gameState = null;
        this.isProcessing = false;
        this.pollInterval = null;
        this.lastStage = null;
        this.displayedActionCount = 0;
        this.lastHandId = null;  // Track last analyzed hand
        this.lastPlayerHand = null;  // Track last rendered player hand to prevent unnecessary re-renders
        
        // Set chat manager session ID
        if (window.chatManager) {
            window.chatManager.setSessionId(this.sessionId);
        }
        
        // Initialize dealer chips to hidden state
        this.initializeDealerChips();
        
        // Initialize notification system
        this.initializeNotifications();
        
        this.initializeEventListeners();
        this.startNewGame();
    }
    
    initializeNotifications() {
        // Create notification container if it doesn't exist
        if (!document.getElementById('notificationContainer')) {
            const container = document.createElement('div');
            container.id = 'notificationContainer';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
    }
    
    showNotification(message, type = 'info', duration = 3000) {
        const container = document.getElementById('notificationContainer');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'polite');
        
        container.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after duration
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }
    
    initializeDealerChips() {
        // Ensure both dealer chips are hidden on initialization
        const playerDealerChip = document.getElementById('playerDealerChip');
        const opponentDealerChip = document.getElementById('opponentDealerChip');
        if (playerDealerChip) playerDealerChip.style.display = 'none';
        if (opponentDealerChip) opponentDealerChip.style.display = 'none';
    }

    initializeEventListeners() {
        // Enable debug logging
        window.DEBUG_POKER = true;

        // Action buttons
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const actionValue = parseInt(e.target.dataset.action);
                const buttonText = e.target.textContent.trim();
                this.handleAction(actionValue);
            });
        });

        // Menu buttons
        document.getElementById('newGameBtn').addEventListener('click', () => {
            this.startNewGame();
        });

        document.getElementById('helpBtn').addEventListener('click', () => {
            this.showHelp();
        });

        // Modal close buttons
        document.getElementById('closeModalBtn').addEventListener('click', () => {
            this.hideModal('gameOverModal');
        });

        document.getElementById('closeHelpBtn').addEventListener('click', () => {
            this.hideModal('helpModal');
        });

        // Close modals on outside click
        document.getElementById('gameOverModal').addEventListener('click', (e) => {
            if (e.target.id === 'gameOverModal') {
                this.hideModal('gameOverModal');
            }
        });

        document.getElementById('helpModal').addEventListener('click', (e) => {
            if (e.target.id === 'helpModal') {
                this.hideModal('helpModal');
            }
        });
    }

    async startNewGame() {
        try {
            this.isProcessing = true;
            this.updateStatus('Starting new game...');
            this.disableAllButtons();

            // Clear action history display
            const historyContainer = document.getElementById('actionHistory');
            if (historyContainer) {
                historyContainer.innerHTML = '';
            }
            this.displayedActionCount = 0;
            this.lastStage = null;
            this.lastPlayerHand = null;  // Reset last hand when starting new game

            const response = await fetch('/api/game/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();
            
            if (!response.ok) {
                const errorMsg = data.error || 'Failed to start game';
                throw new Error(errorMsg);
            }

            this.gameState = data;
            this.isProcessing = false; // Reset processing flag before updating display
            
            // Animate initial card deal
            this.updateDisplay();
            // Trigger card reveal animations for initial deal
            setTimeout(() => {
                this.updatePlayerCards(true);
                this.updateOpponentCards(true);
            }, 100);
            
            this.startPolling();
            
            // Update chat manager session ID
            if (window.chatManager) {
                window.chatManager.setSessionId(this.sessionId);
            }

            // If it's AI's turn, process it automatically
            if (!this.gameState.is_over && this.gameState.current_player === 1) {
                // Use setTimeout to ensure isProcessing flag is fully reset
                setTimeout(() => {
                    this.processAITurn();
                }, 100);
            }
        } catch (error) {
            this.updateStatus('Error starting game. Please try again.');
            this.isProcessing = false;
        }
    }

    async handleAction(actionValue) {

        // Edge case: Already processing
        if (this.isProcessing) {
            this.showNotification('Please wait for current action to complete', 'warning');
            return;
        }

        // Edge case: No game state
        if (!this.gameState) {
            this.showNotification('Game state not available. Please start a new game.', 'error');
            return;
        }

        // Edge case: Not waiting for action
        if (!this.gameState.is_waiting_for_action) {
            this.showNotification('Not your turn yet', 'warning');
            return;
        }

        // Edge case: Game already over
        if (this.gameState.is_over) {
            this.showNotification('Game is already over', 'warning');
            return;
        }

        // Check if action is legal - check both legal_actions and raw_legal_actions (like updateActionButtons does)
        let isLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(actionValue)) ||
                       (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(actionValue));


        // Special case: If RAISE_HALF_POT (2) is clicked but not legal, and RAISE_POT (3) is legal with same label,
        // use RAISE_POT instead (they do the same thing in this context)
        if (!isLegal && actionValue === 2) {
            const raisePotLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(3)) ||
                                 (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(3));
            if (raisePotLegal) {
                // Check if both buttons have the same label (they do the same thing)
                // We need to recalculate button labels to check
                const raised = this.gameState.raised || [0, 0];
                const bigBlind = this.gameState.big_blind || 2;
                const pot = this.gameState.pot || 0;
                const isPreflop = this.gameState.stage === 0;
                const isSmallBlind = this.gameState.button_id === 0;
                const playerRaised = raised[0] || 0;
                const opponentRaised = raised[1] || 0;
                const isFacingBet = opponentRaised > playerRaised;
                const opponentRaisedBB = opponentRaised / bigBlind;
                const bettingLevel = (isPreflop && isFacingBet && opponentRaisedBB <= 4.0) ? 0 : 
                                    (isPreflop && isFacingBet && opponentRaisedBB <= 12.0) ? 1 : 0;
                
                const buttonLabels = this.calculateButtonLabels(
                    isPreflop, isSmallBlind, false, isFacingBet, bettingLevel, 
                    bigBlind, pot, opponentRaised, playerRaised, this.gameState.in_chips || [0, 0]
                );
                if (buttonLabels.raiseHalfPot === buttonLabels.raisePot) {
                    // They have the same label, use RAISE_POT action instead
                    actionValue = 3;
                    isLegal = true;
                }
            }
        }
        
        if (!isLegal) {
            this.updateStatus('Illegal action!');
            this.showNotification('Invalid action selected', 'error');
            return;
        }


        // Action value will be validated on the backend

        try {
            this.isProcessing = true;
            this.disableAllButtons();
            this.updateStatus('Processing your action...');

            // Add animation class to player section for action feedback
            const playerSection = document.querySelector('.player-section-bottom');
            if (playerSection) {
                playerSection.classList.add('action-animating');
                setTimeout(() => playerSection.classList.remove('action-animating'), 1000);
            }


            const response = await fetch('/api/game/action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    action_value: actionValue
                })
            });

            if (!response.ok) {
                const error = await response.json();
                const errorMessage = error.error || 'Failed to process action';
                
                // Check if it's a session not found error
                if (response.status === 404 && errorMessage.includes('Game session not found')) {
                    this.showNotification('Game session expired. Starting new game...', 'warning', 5000);
                    // Restart the game automatically
                    await this.startNewGame();
                    return;
                }
                
                throw new Error(errorMessage);
            }

            this.gameState = await response.json();
            // Check if stage changed (new street dealt)
            const previousStage = this.lastStage;
            const currentStage = this.gameState.stage || 0;
            const stageChanged = (previousStage !== null && previousStage !== currentStage);

            this.updateDisplayWithAnimation('player');

            // If game is over, show results
            if (this.gameState.is_over) {
                this.handleGameEnd();
                return;
            }

            // Wait for animations to complete before processing opponent turn
            // This gives time to see the user's action and its effects
            // If stage changed (new cards dealt), wait longer for card reveal animations
            const animationDelay = stageChanged ? 2200 : 1200;  // Extra 1000ms for card animations
            await new Promise(resolve => setTimeout(resolve, animationDelay));

            // Process AI turn if it's AI's turn
            // Note: We need to temporarily reset isProcessing because processAITurn checks it
            // but we're explicitly calling it after the player's action completes
            if (this.gameState.current_player === 1 && !this.gameState.is_over) {
                // Temporarily reset isProcessing to allow processAITurn to run
                this.isProcessing = false;
                await this.processAITurn();
                // processAITurn will manage isProcessing flag itself
            }
            
            // After processing, update display again to ensure buttons are enabled if it's our turn
            if (this.gameState.current_player === 0 && this.gameState.is_waiting_for_action) {
                this.updateActionButtons();
            }
        } catch (error) {
            console.error('Error processing action:', {
                message: error.message,
                stack: error.stack,
                sessionId: this.sessionId,
                actionValue: actionValue
            });
            const errorMessage = error.message || 'Unknown error occurred';
            this.updateStatus('Error: ' + errorMessage);
            this.showNotification(errorMessage, 'error', 5000);
            
            // If it's a session-related error, offer to restart
            if (errorMessage.includes('session') || errorMessage.includes('Game not found')) {
                if (confirm('Your game session may have expired. Would you like to start a new game?')) {
                    await this.startNewGame();
                    return;
                }
            }
            
            this.updateActionButtons();
        } finally {
            this.isProcessing = false;
        }
    }

    async processAITurn() {
        // Prevent duplicate calls
        if (this.isProcessing) {
            return;
        }
        
        // Double-check it's actually AI's turn
        if (!this.gameState) {
            return;
        }
        
        if (this.gameState.current_player !== 1) {
            return;
        }
        
        if (this.gameState.is_over) {
            return;
        }
        
        this.isProcessing = true;
        this.updateStatus("Opponent is thinking...");
        
        // Add visual thinking indicator
        const opponentSection = document.querySelector('.opponent-section');
        const opponentCards = document.getElementById('opponentCards');
        if (opponentSection) {
            opponentSection.classList.add('thinking-animation');
        }
        if (opponentCards) {
            opponentCards.classList.add('thinking-cards');
        }
        
        // Wait 0.5s before opponent makes decision
        await new Promise(resolve => setTimeout(resolve, 500));

        try {
            const response = await fetch('/api/game/ai-turn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                const error = await response.json();
                const errorMessage = error.error || 'Failed to process AI turn';
                
                // Check if it's a session not found error
                if (response.status === 404 && errorMessage.includes('Game session not found')) {
                    this.showNotification('Game session expired. Starting new game...', 'warning', 5000);
                    // Restart the game automatically
                    await this.startNewGame();
                    return;
                }
                
                throw new Error(errorMessage);
            }

            this.gameState = await response.json();
            
            // Remove thinking animation and add action animation
            if (opponentSection) {
                opponentSection.classList.remove('thinking-animation');
                opponentSection.classList.add('action-animating');
                setTimeout(() => opponentSection.classList.remove('action-animating'), 1000);
            }
            if (opponentCards) {
                opponentCards.classList.remove('thinking-cards');
            }
            
            // Debug logging after AI turn
            console.log('Game state after AI turn:', {
                current_player: this.gameState.current_player,
                is_waiting_for_action: this.gameState.is_waiting_for_action,
                is_over: this.gameState.is_over,
                legal_actions: this.gameState.legal_actions,
                raw_legal_actions: this.gameState.raw_legal_actions
            });
            
            this.updateDisplayWithAnimation('opponent');

            // If game is over, show results
            if (this.gameState.is_over) {
                this.handleGameEnd();
                this.isProcessing = false;
            } else if (this.gameState.current_player === 0 && this.gameState.is_waiting_for_action) {
                // It's our turn again - make sure buttons are enabled
                this.isProcessing = false; // Ensure processing flag is reset
                this.updateActionButtons();
            } else {
                // AI turn completed, reset processing flag
                // (If it's still AI's turn, polling will handle it)
                this.isProcessing = false;
            }
        } catch (error) {
            const errorMessage = error.message || 'Unknown error occurred';
            this.updateStatus('Error processing opponent turn: ' + errorMessage);
            this.showNotification('Error: ' + errorMessage, 'error', 5000);
            
            // If it's a session-related error, offer to restart
            if (errorMessage.includes('session') || errorMessage.includes('Game not found')) {
                if (confirm('Your game session may have expired. Would you like to start a new game?')) {
                    await this.startNewGame();
                    return;
                }
            }
            
            this.isProcessing = false; // Reset on error
        }
    }

    updateDisplay() {
        if (!this.gameState) return;

        // Update cards
        this.updatePlayerCards();
        this.updateCommunityCards();
        this.updateOpponentCards();

        // Update chips and pot
        this.updateChips();
        this.updatePot();
        this.updateStage();

        // Update dealer chip
        this.updateDealerChip();

        // Update bet amounts
        this.updateBetAmounts();

        // Update action history
        this.updateActionHistory();

        // Update turn indicator
        this.updateTurnIndicator();

        // Update status and buttons
        this.updateStatus();
        this.updateActionButtons();
    }

    updateDisplayWithAnimation(actor) {
        if (!this.gameState) return;

        // Update cards with animation
        // Only animate player cards when they are first dealt (container is empty)
        // Don't animate on every turn switch
        const playerCardsContainer = document.getElementById('playerCards');
        const shouldAnimatePlayerCards = playerCardsContainer && playerCardsContainer.children.length === 0;
        this.updatePlayerCards(shouldAnimatePlayerCards);
        this.updateCommunityCards();
        this.updateOpponentCards(actor === 'opponent');

        // Update chips and pot with animation
        this.updateChips(actor);
        this.updatePot();
        this.updateStage();

        // Update dealer chip
        this.updateDealerChip();

        // Update bet amounts with animation
        this.updateBetAmounts(actor);

        // Update action history
        this.updateActionHistory();

        // Update turn indicator
        this.updateTurnIndicator();

        // Update status and buttons
        this.updateStatus();
        this.updateActionButtons();
    }

    updatePlayerCards(animate = false) {
        const container = document.getElementById('playerCards');
        if (!container) return;
        
        if (this.gameState.hand && this.gameState.hand.length > 0) {
            // Create a normalized copy of the current hand for comparison
            const currentHand = [...this.gameState.hand].sort().join(',');
            
            // Only update if cards actually changed or if we need to animate
            const cardsChanged = this.lastPlayerHand !== currentHand;
            const isEmpty = container.children.length === 0;
            
            if (cardsChanged || isEmpty || animate) {
                // Update the stored hand value
                this.lastPlayerHand = currentHand;
                
                container.innerHTML = '';
                this.gameState.hand.forEach((card, index) => {
                    const cardElement = this.createCardElement(card);
                    if (animate) {
                        cardElement.classList.add('card-reveal');
                        cardElement.style.animationDelay = `${index * 0.15}s`;
                    }
                    container.appendChild(cardElement);
                });
            }
        } else {
            // Clear stored hand when hand is empty
            this.lastPlayerHand = null;
            // Only clear if there are cards currently displayed
            if (container.children.length > 0) {
                container.innerHTML = '';
            }
        }
    }

    updateCommunityCards() {
        const container = document.getElementById('communityCards');
        const existingCount = container.children.length;
        const newCount = this.gameState.public_cards ? this.gameState.public_cards.length : 0;
        const currentStage = this.gameState.stage || 0;

        // Check if stage changed (round ended) - this indicates flop/turn/river is being dealt
        // lastStage is updated in updateBetAmounts() which is called after this, so it still has previous value
        const stageChanged = (this.lastStage !== null && this.lastStage !== currentStage);

        // Only update if new cards are added AND the game is not over
        if (newCount > existingCount && !this.gameState.is_over) {
            if (this.gameState.public_cards && this.gameState.public_cards.length > 0) {
                // If stage changed (round ended), delay card reveal by 1.0s
                // Otherwise show immediately (e.g., initial deal)
                const delay = stageChanged ? 1000 : 0;
                
                // Store reference to current gameState to avoid closure issues
                const publicCards = this.gameState.public_cards;
                const startIndex = existingCount;
                
                setTimeout(() => {
                    // Re-check container state in case it was reset during delay
                    const currentExistingCount = container.children.length;
                    const currentNewCount = this.gameState.public_cards ? this.gameState.public_cards.length : 0;
                    
                    // Only add cards if they're still new and container hasn't been reset
                    if (currentNewCount > currentExistingCount && currentExistingCount >= startIndex) {
                        // Add only new cards with animation
                        publicCards.forEach((card, index) => {
                            if (index >= currentExistingCount) {
                                const cardElement = this.createCardElement(card);
                                cardElement.classList.add('card-reveal');
                                cardElement.style.animationDelay = `${(index - currentExistingCount) * 0.2}s`;
                                container.appendChild(cardElement);
                            }
                        });
                    }
                }, delay);
            }
        } else if (newCount < existingCount || !this.gameState.public_cards) {
            // Reset if cards decreased (new hand)
            container.innerHTML = '';
            if (this.gameState.public_cards && this.gameState.public_cards.length > 0) {
                this.gameState.public_cards.forEach((card, index) => {
                    const cardElement = this.createCardElement(card);
                    container.appendChild(cardElement);
                });
            }
        }
    }

    updateOpponentCards(animate = false) {
        const container = document.getElementById('opponentCards');
        const wasOver = container.querySelector('.card:not(.back)') !== null;
        const isOver = this.gameState.is_over && this.gameState.opponent_hand;
        
        container.innerHTML = '';

        // Show opponent cards if game is over
        if (isOver && this.gameState.opponent_hand) {
            this.gameState.opponent_hand.forEach((card, index) => {
                const cardElement = this.createCardElement(card);
                if (animate && !wasOver) {
                    // Animate reveal at showdown
                    cardElement.classList.add('card-reveal', 'card-flip');
                    cardElement.style.animationDelay = `${index * 0.3}s`;
                }
                container.appendChild(cardElement);
            });
        } else {
            // Show card backs
            for (let i = 0; i < 2; i++) {
                const cardElement = document.createElement('div');
                cardElement.className = 'card back';
                if (animate) {
                    cardElement.classList.add('card-pulse');
                }
                container.appendChild(cardElement);
            }
        }
    }

    createCardElement(cardString) {
        const card = document.createElement('div');
        card.className = 'card';

        // Format card display
        const formatted = this.formatCard(cardString);
        card.textContent = formatted;

        // Determine color
        if (formatted.includes('♥') || formatted.includes('♦')) {
            card.classList.add('red');
        } else {
            card.classList.add('black');
        }

        return card;
    }

    formatCard(cardString) {
        // Ensure cardString is a string
        if (typeof cardString !== 'string') {
            return '??';
        }

        if (!cardString || cardString === '??') {
            return '??';
        }

        // Convert from format like 'SJ' to '♠J'
        const suitMap = {
            'S': '♠',
            'H': '♥',
            'D': '♦',
            'C': '♣'
        };

        if (cardString.length >= 2) {
            const suit = suitMap[cardString[0]] || cardString[0];
            const rank = cardString.substring(1);
            return suit + rank;
        }

        return cardString;
    }

    updateChips(actor = null) {
        if (this.gameState.stakes && this.gameState.stakes.length >= 2) {
            const bigBlind = this.gameState.big_blind || 2;
            const playerBB = (this.gameState.stakes[0] / bigBlind).toFixed(1);
            const opponentBB = (this.gameState.stakes[1] / bigBlind).toFixed(1);
            
            const playerChipsEl = document.getElementById('playerChips');
            const opponentChipsEl = document.getElementById('opponentChips');
            
            // Check if values changed to trigger animation
            if (playerChipsEl) {
                const oldValue = playerChipsEl.textContent;
                const newValue = `${playerBB} BB`;
                if (oldValue !== newValue) {
                    playerChipsEl.textContent = newValue;
                if (actor === 'player') {
                    playerChipsEl.classList.add('chip-update');
                    setTimeout(() => playerChipsEl.classList.remove('chip-update'), 1000);
                }
                }
            }
            
            if (opponentChipsEl) {
                const oldValue = opponentChipsEl.textContent;
                const newValue = `${opponentBB} BB`;
                if (oldValue !== newValue) {
                    opponentChipsEl.textContent = newValue;
                if (actor === 'opponent') {
                    opponentChipsEl.classList.add('chip-update');
                    setTimeout(() => opponentChipsEl.classList.remove('chip-update'), 1000);
                }
                }
            }
        }
    }

    updatePot() {
        if (this.gameState.pot !== undefined) {
            const bigBlind = this.gameState.big_blind || 2;
            const pot = this.gameState.pot || 0;
            
            // Backend now calculates pot accurately using centralized pot calculator
            // Pot includes all bets (matched + unmatched) during betting rounds
            // Simply convert to BB for display
            const potBB = (pot / bigBlind).toFixed(1);
            
            const potEl = document.getElementById('potDisplay');
            if (potEl) {
                const oldValue = potEl.textContent;
                const newValue = `Pot: ${potBB} BB`;
                if (oldValue !== newValue) {
                    potEl.textContent = newValue;
                    potEl.classList.add('pot-update');
                    setTimeout(() => potEl.classList.remove('pot-update'), 1000);
                }
            }
        }
    }

    updateStage() {
        const stageNames = ['Preflop', 'Flop', 'Turn', 'River', 'Showdown'];
        const stage = this.gameState.stage || 0;
        const stageName = stageNames[Math.min(stage, stageNames.length - 1)];
        document.getElementById('stageDisplay').textContent = stageName;
    }

    updateDealerChip() {
        const playerDealerChip = document.getElementById('playerDealerChip');
        const opponentDealerChip = document.getElementById('opponentDealerChip');
        
        // Always hide both chips first - this is critical
        if (playerDealerChip) {
            playerDealerChip.style.display = 'none';
        }
        if (opponentDealerChip) {
            opponentDealerChip.style.display = 'none';
        }
        
        // In HUNL, button is on the small blind position
        // Use button_id if available, otherwise calculate from dealer_id
        let buttonId = this.gameState?.button_id;
        
        // If button_id is not available, try to calculate from dealer_id
        if (buttonId === null || buttonId === undefined || buttonId === '') {
            const dealerId = this.gameState?.dealer_id;
            if (dealerId !== null && dealerId !== undefined && dealerId !== '') {
                // Button = small blind = (dealer_id + 1) % num_players
                // In HUNL with 2 players: button alternates between 0 and 1
                buttonId = (dealerId + 1) % 2;
            } else {
                // If we can't determine button, don't show any chip
                return;
            }
        }
        
        // Convert to number if it's a string
        buttonId = parseInt(buttonId, 10);
        
        // Show chip only for the button player (small blind in HUNL)
        if (!isNaN(buttonId)) {
            if (buttonId === 0 && playerDealerChip) {
                playerDealerChip.style.display = 'block';
            } else if (buttonId === 1 && opponentDealerChip) {
                opponentDealerChip.style.display = 'block';
            }
        }
    }

    updateBetAmounts(actor = null) {
        if (!this.gameState) return;

        const bigBlind = this.gameState.big_blind || 2;
        const raised = this.gameState.raised || [0, 0];
        const in_chips = this.gameState.in_chips || [0, 0];
        
        // Check if stage changed (new round started) - bet amounts should be cleared
        const currentStage = this.gameState.stage || 0;
        const stageChanged = (this.lastStage !== null && this.lastStage !== currentStage);
        this.lastStage = currentStage;

        // Calculate bet amounts to display
        // Show the actual amount each player has bet THIS ROUND (from raised array)
        let playerBet = raised[0] || 0;
        let opponentBet = raised[1] || 0;
        
        // Note: Bet amounts are now reliably tracked via raised array from backend
        // No need for fallback inference from action history
        
        // Calculate unmatched amounts (for pot calculation)
        const playerUnmatched = Math.max(0, playerBet - opponentBet);
        const opponentUnmatched = Math.max(0, opponentBet - playerBet);

        // Update player bet amount (show actual bet amount, not just unmatched)
        const playerBetEl = document.getElementById('playerBetAmount');
        if (playerBetEl) {
            const betValueEl = playerBetEl.querySelector('.bet-value');
            // Show bet amount if player has bet this round, and stage hasn't just changed
            if (playerBet > 0 && !stageChanged && betValueEl) {
                const betBB = (playerBet / bigBlind).toFixed(1);
                const wasHidden = !playerBetEl.style.display || playerBetEl.style.display === 'none' || playerBetEl.style.opacity === '0';
                betValueEl.textContent = `${betBB} BB`;
                playerBetEl.style.display = 'flex';
                playerBetEl.style.opacity = '1';
                playerBetEl.style.visibility = 'visible';
                if (actor === 'player' && wasHidden) {
                    playerBetEl.classList.add('bet-appear');
                    setTimeout(() => playerBetEl.classList.remove('bet-appear'), 1000);
                } else if (actor === 'player') {
                    playerBetEl.classList.add('bet-update');
                    setTimeout(() => playerBetEl.classList.remove('bet-update'), 1000);
                }
            } else {
                // Hide when stage changes (new round) or no bet (but keep space reserved)
                if (betValueEl) {
                    betValueEl.textContent = '';
                }
                playerBetEl.style.opacity = '0';
                playerBetEl.style.visibility = 'hidden';
            }
        }

        // Update opponent bet amount (show actual bet amount, not just unmatched)
        const opponentBetEl = document.getElementById('opponentBetAmount');
        if (opponentBetEl) {
            const betValueEl = opponentBetEl.querySelector('.bet-value');
            // Show bet amount if opponent has bet this round, and stage hasn't just changed
            if (opponentBet > 0 && !stageChanged && betValueEl) {
                const betBB = (opponentBet / bigBlind).toFixed(1);
                const wasHidden = !opponentBetEl.style.display || opponentBetEl.style.display === 'none' || opponentBetEl.style.opacity === '0';
                betValueEl.textContent = `${betBB} BB`;
                opponentBetEl.style.display = 'flex';
                opponentBetEl.style.opacity = '1';
                opponentBetEl.style.visibility = 'visible';
                if (actor === 'opponent' && wasHidden) {
                    opponentBetEl.classList.add('bet-appear');
                    setTimeout(() => opponentBetEl.classList.remove('bet-appear'), 1000);
                } else if (actor === 'opponent') {
                    opponentBetEl.classList.add('bet-update');
                    setTimeout(() => opponentBetEl.classList.remove('bet-update'), 1000);
                }
            } else {
                // Hide when stage changes (new round) or no bet (but keep space reserved)
                if (betValueEl) {
                    betValueEl.textContent = '';
                }
                opponentBetEl.style.opacity = '0';
                opponentBetEl.style.visibility = 'hidden';
            }
        }
    }

    updateActionHistory() {
        // Use new event-based history (hand_events) if available, fallback to legacy action_history
        const handEvents = this.gameState?.hand_events || [];
        const legacyHistory = this.gameState?.action_history || [];
        
        const historyContainer = document.getElementById('actionHistory');
        if (!historyContainer) return;

        // Use event-based history if available
        if (handEvents.length > 0) {
            // Only update if there are new events
            if (handEvents.length <= this.displayedActionCount) return;

            // Add new events
            for (let i = this.displayedActionCount; i < handEvents.length; i++) {
                const event = handEvents[i];
                
                // Show action label above cards for action and blind events
                if ((event.kind === 'action' || event.kind === 'blind') && event.player_idx !== null) {
                    this.showActionLabelOnCards(event);
                }
                
                const actionEntry = document.createElement('div');
                
                // Handle different event types
                if (event.kind === 'community') {
                    // Community cards (Flop/Turn/River)
                    actionEntry.className = 'action-entry community-cards-entry';
                    const cardsText = event.cards.map(card => this.formatCard(card)).join(' ');
                    actionEntry.innerHTML = `
                        <span class="community-cards-label">${event.label}:</span>
                        <span class="community-cards">${cardsText}</span>
                    `;
                } else if (event.kind === 'blind') {
                    // Blind posting
                    actionEntry.className = `action-entry blind-entry ${event.player_idx === 0 ? 'player-action' : 'opponent-action'}`;
                    const playerName = event.player_idx === 0 ? 'You' : 'Opponent';
                    const bigBlind = this.gameState.big_blind || 2;
                    const blindBB = event.amount ? (event.amount / bigBlind).toFixed(1) : '0';
                    actionEntry.innerHTML = `
                        <span class="player-name">${playerName}:</span>
                        <span class="blind-label">${event.label}</span>
                        <span class="bet-amount">(${blindBB} BB)</span>
                    `;
                } else if (event.kind === 'action') {
                    // Player action
                    actionEntry.className = `action-entry ${event.player_idx === 0 ? 'player-action' : 'opponent-action'}`;
                    const playerName = event.player_idx === 0 ? 'You' : 'Opponent';
                    actionEntry.innerHTML = `
                        <span class="player-name">${playerName}:</span>
                        <span class="action-name">${event.label}</span>
                    `;
                } else if (event.kind === 'win') {
                    // Hand conclusion
                    actionEntry.className = `action-entry win-entry ${event.player_idx === 0 ? 'player-action' : 'opponent-action'}`;
                    const playerName = event.player_idx === 0 ? 'You' : 'Opponent';
                    actionEntry.innerHTML = `
                        <span class="player-name">${playerName}:</span>
                        <span class="win-label">${event.label}</span>
                    `;
                }
                
                historyContainer.appendChild(actionEntry);
            }

            // Scroll to bottom to show latest action
            historyContainer.scrollTop = historyContainer.scrollHeight;
            this.displayedActionCount = handEvents.length;

            // Limit displayed actions to last 20
            while (historyContainer.children.length > 20) {
                historyContainer.removeChild(historyContainer.firstChild);
            }
        } else if (legacyHistory.length > 0) {
            // Fallback to legacy format for backward compatibility
            if (legacyHistory.length <= this.displayedActionCount) return;
            
            // Process legacy format (simplified - just show basic info)
            for (let i = this.displayedActionCount; i < legacyHistory.length; i++) {
                const action = legacyHistory[i];
                const actionEntry = document.createElement('div');
                
                if (action.type === 'community_cards') {
                    actionEntry.className = 'action-entry community-cards-entry';
                    const cardsText = (action.all_cards || []).map(card => this.formatCard(card)).join(' ');
                    actionEntry.innerHTML = `
                        <span class="community-cards-label">${action.stage}:</span>
                        <span class="community-cards">${cardsText}</span>
                    `;
                } else if (action.type === 'blind') {
                    actionEntry.className = `action-entry blind-entry ${action.player_id === 0 ? 'player-action' : 'opponent-action'}`;
                    const bigBlind = this.gameState.big_blind || 2;
                    const blindBB = action.amount ? (action.amount / bigBlind).toFixed(1) : '0';
                    const blindLabel = action.blind_type === 'small' ? 'Posts Small Blind' : 'Posts Big Blind';
                    actionEntry.innerHTML = `
                        <span class="player-name">${action.player_name}:</span>
                        <span class="blind-label">${blindLabel}</span>
                        <span class="bet-amount">(${blindBB} BB)</span>
                    `;
                } else {
                    actionEntry.className = `action-entry ${action.player_id === 0 ? 'player-action' : 'opponent-action'}`;
                    actionEntry.innerHTML = `
                        <span class="player-name">${action.player_name}:</span>
                        <span class="action-name">${action.action || 'Action'}</span>
                    `;
                }
                
                historyContainer.appendChild(actionEntry);
            }
            
            historyContainer.scrollTop = historyContainer.scrollHeight;
            this.displayedActionCount = legacyHistory.length;
            
            while (historyContainer.children.length > 20) {
                historyContainer.removeChild(historyContainer.firstChild);
            }
        }
    }

    showActionLabelOnCards(event) {
        // Show action label above the correct player's cards based on event
        if (!event || event.player_idx === null || event.player_idx === undefined) return;
        
        const isOpponent = event.player_idx === 1;
        const cardsContainer = isOpponent 
            ? document.getElementById('opponentCards')
            : document.getElementById('playerCards');
        
        if (!cardsContainer) return;
        
        // Get the cards wrapper to position the action text
        const cardsWrapper = cardsContainer.closest('.cards-wrapper');
        if (!cardsWrapper) return;
        
        // Remove any existing action text overlay in this wrapper
        const existingOverlay = cardsWrapper.querySelector('.action-text-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        // Create action text overlay
        const overlay = document.createElement('div');
        overlay.className = 'action-text-overlay';
        
        // Use the label directly from the event (already formatted correctly)
        overlay.textContent = event.label || '';
        
        // Position in cards wrapper (which is already position: relative)
        cardsWrapper.appendChild(overlay);
        
        // Trigger animation
        setTimeout(() => {
            overlay.classList.add('show');
        }, 10);
        
        // Remove after 1.0 seconds
        setTimeout(() => {
            overlay.classList.remove('show');
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.parentNode.removeChild(overlay);
                }
            }, 300); // Wait for fade out animation
        }, 1000);
    }

    updateTurnIndicator() {
        if (!this.gameState || this.gameState.is_over) {
            // Remove turn indicators when game is over
            document.querySelectorAll('.player-section').forEach(section => {
                section.classList.remove('active-turn');
            });
            return;
        }

        const playerSection = document.querySelector('.player-section-bottom');
        const opponentSection = document.querySelector('.opponent-section');

        // Remove active-turn class from both sections first
        if (playerSection) playerSection.classList.remove('active-turn');
        if (opponentSection) opponentSection.classList.remove('active-turn');

        // Add active-turn class to the current player's section
        if (this.gameState.current_player === 0 && playerSection) {
            playerSection.classList.add('active-turn');
        } else if (this.gameState.current_player === 1 && opponentSection) {
            opponentSection.classList.add('active-turn');
        }
    }

    updateStatus(message) {
        const statusEl = document.getElementById('statusDisplay');
        
        if (message) {
            statusEl.textContent = message;
            return;
        }

        if (!this.gameState) {
            statusEl.textContent = 'Waiting to start...';
            return;
        }

        if (this.gameState.is_over) {
            statusEl.textContent = 'Game Over';
            return;
        }

        if (this.gameState.current_player === 0) {
            statusEl.textContent = 'Your Turn';
        } else {
            statusEl.textContent = "Opponent's Turn";
        }
    }

    async updateActionButtons() {
        if (!this.gameState || this.gameState.is_over) {
            this.disableAllButtons();
            return;
        }

        const bigBlind = this.gameState.big_blind || 2;
        const pot = this.gameState.pot || 0;
        const raised = this.gameState.raised || [0, 0];
        const stage = this.gameState.stage || 0;
        const buttonId = this.gameState.button_id;
        const actionHistory = this.gameState.action_history || [];
        const inChips = this.gameState.in_chips || [0, 0];
        
        const isPreflop = stage === 0;
        const isSmallBlind = buttonId === 0; // Player 0 is small blind if button_id is 0
        
        // Determine if it's "Check" or "Call"
        const playerRaised = raised[0] || 0;
        const opponentRaised = raised[1] || 0;
        
        // Determine if facing a bet using ActionLabeling-compatible logic
        const epsilon = 0.01;
        let isFacingBet = false;

        if (isPreflop) {
            // Preflop: If opponent has raised more than player, player is facing a bet
            // Special case: SB always faces BB's bet preflop (SB must call BB's bet)
            if (isSmallBlind && opponentRaised >= bigBlind * 0.9 && playerRaised < bigBlind * 0.9) {
                // SB facing BB's bet (blind post scenario)
                isFacingBet = true;
            } else {
                // Standard logic: facing bet if opponent raised more than player
                isFacingBet = (opponentRaised - playerRaised) > epsilon;
            }
        } else {
            // Postflop: facing bet if raised amounts differ
            isFacingBet = Math.abs(opponentRaised - playerRaised) > epsilon;
        }
        
        // Determine if first to act using ActionLabeling-compatible logic
        let isFirstToAct = false;

        if (isPreflop) {
            // First to act preflop if not facing a bet
            isFirstToAct = !isFacingBet;
        } else {
            // Postflop: first to act if both players have same raised amount (new betting round)
            isFirstToAct = Math.abs(playerRaised - opponentRaised) <= epsilon;
        }
        
        // Determine if it's "Check" or "Call"
        // "Call" if facing a bet (opponent has bet more than player)
        // "Check" if not facing a bet (player has matched or bet more than opponent)
        const isCheck = !isFacingBet;
        const checkCallText = isCheck ? 'Check' : 'Call';
        
        // Debug logging (can be removed in production)
        if (window.DEBUG_POKER) {
            console.log('Action button context:', {
                playerRaised,
                opponentRaised,
                diff: opponentRaised - playerRaised,
                isFacingBet,
                isCheck,
                checkCallText,
                isPreflop,
                isSmallBlind,
                bigBlind,
                actionHistory: actionHistory.slice(-3)
            });
        }
        
        // Determine betting level for preflop
        let bettingLevel = 0; // 0 = open, 1 = 3-bet, 2 = 4-bet, etc.
        if (isPreflop && isFacingBet) {
            // Determine betting level based on opponent's raise amount, not pot size or raise count
            // This is more accurate for determining what we're facing
            const opponentRaisedBB = opponentRaised / bigBlind;
            
            if (opponentRaisedBB <= 1.1) {
                // Opponent just posted big blind (or checked) - shouldn't happen when facing bet
                bettingLevel = 0;
            } else if (opponentRaisedBB <= 4.0) {
                // Facing a button open (3BB) - we're making a 3-bet
                bettingLevel = 0; // Open level (we're facing an open, making a 3-bet)
            } else if (opponentRaisedBB <= 12.0) {
                // Facing a 3-bet (4-12BB) - we're making a 4-bet
                bettingLevel = 1; // 3-bet level (we're facing a 3-bet, making a 4-bet)
            } else {
                // Facing a 4-bet or higher (12BB+) - no 5-betting allowed
                bettingLevel = 2; // 4-bet level (we're facing a 4-bet, no raise options)
            }
        }
        
        // Get button labels from backend API for consistency - API is authoritative
        let buttonLabels = {
            raiseHalfPot: 'Raise ½ Pot',
            raisePot: 'Raise Pot',
            showRaiseHalfPot: true,
            showRaisePot: true,
            checkCall: checkCallText
        };

        // Always try to fetch from API first - this should be the primary source
        let apiSuccess = false;
        try {
            console.log('Fetching button labels with game state:', {
                stage: this.gameState.stage,
                button_id: this.gameState.button_id,
                raised: this.gameState.raised,
                pot: this.gameState.pot,
                current_player: this.gameState.current_player
            });
            const response = await fetch('/api/game/button-labels', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (response.ok) {
                const apiLabels = await response.json();
                buttonLabels = {
                    raiseHalfPot: apiLabels.raiseHalfPot || buttonLabels.raiseHalfPot,
                    raisePot: apiLabels.raisePot || buttonLabels.raisePot,
                    showRaiseHalfPot: apiLabels.showRaiseHalfPot !== undefined ? apiLabels.showRaiseHalfPot : buttonLabels.showRaiseHalfPot,
                    showRaisePot: apiLabels.showRaisePot !== undefined ? apiLabels.showRaisePot : buttonLabels.showRaisePot,
                    checkCall: apiLabels.checkCall || checkCallText
                };
                apiSuccess = true;
            } else {
                const errorText = await response.text();
            }
        } catch (error) {
        }

        // Only use local calculation as fallback if API completely failed
        if (!apiSuccess) {

            // Use ActionLabeling-compatible logic for fallback
            // Reconstruct context similar to ActionLabeling.get_context_from_state
            const epsilon = 0.01;
            let localIsFacingBet = false;
            let localIsFirstToAct = false;
            let localBettingLevel = 0;

            if (isPreflop) {
                // Preflop: If opponent has raised more than player, player is facing a bet
                // Special case: SB always faces BB's bet preflop (SB must call BB's bet)
                if (isSmallBlind && opponentRaised >= bigBlind * 0.9 && playerRaised < bigBlind * 0.9) {
                    // SB facing BB's bet (blind post scenario)
                    localIsFacingBet = true;
                } else {
                    // Standard logic: facing bet if opponent raised more than player
                    localIsFacingBet = (opponentRaised - playerRaised) > epsilon;
                }

                // First to act preflop if not facing a bet
                localIsFirstToAct = !localIsFacingBet;
            } else {
                // Postflop: facing bet if raised amounts differ
                localIsFacingBet = Math.abs(opponentRaised - playerRaised) > epsilon;
                localIsFirstToAct = playerRaised === opponentRaised;
            }

            // Determine betting level for preflop
            if (isPreflop && localIsFacingBet) {
                const opponentRaisedBB = opponentRaised / bigBlind;
                if (opponentRaisedBB <= 4.0) {
                    localBettingLevel = 0;  // Facing open
                } else if (opponentRaisedBB <= 12.0) {
                    localBettingLevel = 1;  // Facing 3-bet
                } else {
                    localBettingLevel = 2;  // Facing 4-bet+
                }
            }

            console.log('Button label calculation context:', {
                isPreflop, isSmallBlind, localIsFirstToAct, localIsFacingBet, localBettingLevel,
                bigBlind, pot, opponentRaised, playerRaised
            });

            buttonLabels = this.calculateButtonLabels(
                isPreflop, isSmallBlind, localIsFirstToAct, localIsFacingBet,
                localBettingLevel, bigBlind, pot, opponentRaised, playerRaised, inChips
            );
            buttonLabels.checkCall = localIsFacingBet ? 'Call' : 'Check';
        }

        const buttons = document.querySelectorAll('.action-btn');
        
        buttons.forEach((btn) => {
            const actionValue = parseInt(btn.dataset.action);
            // Check both legal_actions and raw_legal_actions
            const isLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(actionValue)) ||
                           (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(actionValue));

            // Debug logging for legal actions
            if (window.DEBUG_POKER && actionValue === 0) {  // Only log once per update
                console.log('Legal actions check:', {
                    actionValue,
                    legal_actions: this.gameState.legal_actions,
                    raw_legal_actions: this.gameState.raw_legal_actions,
                    isLegal,
                    isWaiting: this.gameState.is_waiting_for_action,
                    isMyTurn: this.gameState.current_player === 0
                });
            }
            const isWaiting = this.gameState.is_waiting_for_action;
            const isMyTurn = this.gameState.current_player === 0;
            const isGameActive = !this.gameState.is_over;

            // Enable/disable button - be more permissive for debugging
            // Allow actions if legal and game is active, regardless of whose turn it is
            const shouldEnable = isLegal && isGameActive;
            btn.disabled = !shouldEnable;

            // Debug logging for button state
            if (window.DEBUG_POKER && actionValue === 0) {  // Only log once per update
                console.log('Button state update:', {
                    actionValue,
                    current_player: this.gameState.current_player,
                    is_waiting_for_action: this.gameState.is_waiting_for_action,
                    is_over: this.gameState.is_over,
                    isLegal,
                    isGameActive,
                    shouldEnable,
                    disabled: btn.disabled
                });
            }

            // Update button text based on context
            if (actionValue === 1) { // CHECK_CALL
                btn.textContent = buttonLabels.checkCall || checkCallText;
                btn.setAttribute('aria-label', `${checkCallText} - ${isCheck ? 'Pass without betting' : 'Match the current bet'}`);
            } else if (actionValue === 2) { // RAISE_HALF_POT
                btn.textContent = buttonLabels.raiseHalfPot;
                // Special case: If both buttons have the same label and RAISE_POT is legal but RAISE_HALF_POT is not,
                // we should hide RAISE_HALF_POT and show RAISE_POT instead
                const raisePotLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(3)) ||
                                     (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(3));
                if (!buttonLabels.showRaiseHalfPot ||
                    (buttonLabels.raiseHalfPot === buttonLabels.raisePot && !isLegal && raisePotLegal)) {
                    btn.style.display = 'none';
                } else {
                    btn.style.display = '';
                }
            } else if (actionValue === 3) { // RAISE_POT
                btn.textContent = buttonLabels.raisePot;
                // Special case: If both buttons have the same label, show RAISE_POT if RAISE_HALF_POT is not legal
                const raiseHalfPotLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(2)) ||
                                         (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(2));
                if (!buttonLabels.showRaisePot) {
                    btn.style.display = 'none';
                } else if (buttonLabels.raiseHalfPot === buttonLabels.raisePot && !raiseHalfPotLegal && isLegal) {
                    // Both have same label, RAISE_HALF_POT is not legal, but RAISE_POT is - show it
                    btn.style.display = '';
                } else {
                    btn.style.display = buttonLabels.showRaisePot ? '' : 'none';
                }
            }

            // Enable/disable button based on legality and game state
            // For raise buttons, also check if they should be visible
            // Fold (0) and All-In (4) should always be visible if legal
            let shouldBeVisible = (actionValue === 0) || // Fold - always visible if legal
                                   (actionValue === 1) || // Check/Call - always visible if legal
                                   (actionValue === 4) || // All-In - always visible if legal
                                   (actionValue === 2 && buttonLabels.showRaiseHalfPot) ||
                                   (actionValue === 3 && buttonLabels.showRaisePot);
            
            // Special case: If both raise buttons have the same label (e.g., "3-bet to 10 BB")
            // and one action is legal but the other is not, enable the legal one
            if (buttonLabels.raiseHalfPot === buttonLabels.raisePot) {
                const raiseHalfPotLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(2)) ||
                                         (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(2));
                const raisePotLegal = (this.gameState.legal_actions && this.gameState.legal_actions.includes(3)) ||
                                     (this.gameState.raw_legal_actions && this.gameState.raw_legal_actions.includes(3));
                
                if (actionValue === 2 && !raiseHalfPotLegal && raisePotLegal) {
                    // RAISE_HALF_POT is not legal, but RAISE_POT is - treat as legal for display purposes
                    shouldBeVisible = true;
                    isLegal = true; // Override for this case
                } else if (actionValue === 3 && !raisePotLegal && raiseHalfPotLegal) {
                    // RAISE_POT is not legal, but RAISE_HALF_POT is - treat as legal for display purposes
                    shouldBeVisible = true;
                    isLegal = true; // Override for this case
                }
            }
            
            if (isLegal && isWaiting && isMyTurn && !this.isProcessing && shouldBeVisible) {
                btn.disabled = false;
            } else {
                btn.disabled = true;
            }
        });
    }
    
    calculateButtonLabels(isPreflop, isSmallBlind, isFirstToAct, isFacingBet, bettingLevel, bigBlind, pot, opponentRaised, playerRaised, inChips) {
        const labels = {
            raiseHalfPot: 'Raise ½ Pot',
            raisePot: 'Raise Pot',
            showRaiseHalfPot: true,
            showRaisePot: true
        };
        
        if (isPreflop) {
            // PREFLOP LOGIC - All preflop decisions should be in BB
            // IMPORTANT: Check isFacingBet FIRST - if facing a bet, we're not "first to act" for betting purposes
            if (isFacingBet) {
                // Facing a bet (big blind vs open, or facing 3-bet/4-bet)
                if (bettingLevel === 0) {
                    // Facing an open (2.5-3BB), show 3-bet option
                    labels.raiseHalfPot = '3-bet to 10 BB';
                    labels.raisePot = '3-bet to 10 BB';
                    labels.showRaisePot = false; // PREFLOP: Only show ONE bet size option
                } else if (bettingLevel === 1) {
                    // Facing a 3-bet, show 4-bet option (only one size)
                    labels.raiseHalfPot = '4-bet to 25 BB';
                    labels.raisePot = '4-bet to 25 BB';
                    labels.showRaisePot = false; // PREFLOP: Only show ONE bet size option
                } else {
                    // Facing 4-bet or higher, show one raise size
                    const betToCall = opponentRaised - playerRaised;
                    const raise25xTotal = betToCall * 2.5 + betToCall;
                    const raise25xBB = Math.round(raise25xTotal / bigBlind);
                    labels.raiseHalfPot = `Raise to ${raise25xBB} BB`;
                    labels.raisePot = `Raise to ${raise25xBB} BB`;
                    labels.showRaisePot = false; // PREFLOP: Only show ONE bet size option
                }
            } else if (isSmallBlind && isFirstToAct) {
                // Small blind opening (unopened pot, only blinds posted)
                // Only show this if NOT facing a bet
                labels.raiseHalfPot = 'Raise to 3 BB';
                labels.raisePot = 'Raise Pot';
                labels.showRaisePot = false; // PREFLOP: Only show ONE bet size option
            } else {
                // Big blind not facing a bet (can still raise) - show BB amounts
                // This happens when small blind just called, big blind can check or raise
                // Standard preflop raise size: 3BB only
                labels.raiseHalfPot = 'Raise to 3 BB';
                labels.raisePot = 'Raise Pot';
                labels.showRaisePot = false; // PREFLOP: Only show ONE bet size option
            }
        } else {
            // POSTFLOP LOGIC - Always show TWO bet size options
            if (isFirstToAct) {
                // First to act (continuation bet, value bet, etc.)
                // Show most common sizing options
                labels.raiseHalfPot = 'Bet ½ Pot';
                labels.raisePot = 'Bet ⅔ Pot';
                // POSTFLOP: Always show TWO bet size options
            } else if (isFacingBet) {
                // Facing a bet (raise sizing based on opponent's bet)
                const betToCall = opponentRaised - playerRaised;

                // Calculate raise sizes: 2.5x and 3x the bet (standard raise sizing)
                // Total raise = bet to call * multiplier + bet to call
                const raise25xTotal = betToCall * 2.5 + betToCall;
                const raise3xTotal = betToCall * 3 + betToCall;

                // Convert to BB and round
                const raise25xBB = Math.round(raise25xTotal / bigBlind);
                const raise3xBB = Math.round(raise3xTotal / bigBlind);

                labels.raiseHalfPot = `Raise to ${raise25xBB} BB`;
                labels.raisePot = `Raise to ${raise3xBB} BB`;
                // POSTFLOP: Always show TWO bet size options
            } else {
                // Other postflop situations (shouldn't happen often)
                // Ensure consistency with first to act - show two bet sizes
                labels.raiseHalfPot = 'Bet ½ Pot';
                labels.raisePot = 'Bet ⅔ Pot';
                // POSTFLOP: Always show TWO bet size options
            }
        }
        
        return labels;
    }

    disableAllButtons() {
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.disabled = true;
        });
    }

    handleGameEnd() {
        this.stopPolling();
        this.disableAllButtons();

        const bigBlind = this.gameState.big_blind || 2;
        const potBB = this.gameState.pot.toFixed(1);
        const playerBB = (this.gameState.stakes[0] / bigBlind).toFixed(1);
        const opponentBB = (this.gameState.stakes[1] / bigBlind).toFixed(1);

        let message = '';
        if (this.gameState.payoffs) {
            if (this.gameState.payoffs[0] > 0) {
                message = '🎉 You Win! 🎉\n\n';
            } else if (this.gameState.payoffs[0] < 0) {
                message = 'Opponent Wins\n\n';
            } else {
                message = "It's a Tie!\n\n";
            }
        }

        message += `Final Pot: ${potBB} BB\n`;
        if (this.gameState.stakes && this.gameState.stakes.length >= 2) {
            message += `Your Chips: ${playerBB} BB\n`;
            message += `Opponent Chips: ${opponentBB} BB`;
        }

        document.getElementById('gameOverTitle').textContent = 'Game Over';
        document.getElementById('gameOverMessage').textContent = message;
        this.showModal('gameOverModal');

        // Hand analysis popup removed - no longer triggered after each hand
    }
    
    async analyzeHand() {
        try {
            // Get complete hand history from backend
            const historyResponse = await fetch(`/api/coach/get-hand-history?session_id=${this.sessionId}`);
            let handHistory = [];
            
            if (historyResponse.ok) {
                const handData = await historyResponse.json();
                handHistory = handData.decisions || [];
            } else {
                // Fallback to action history from game state
                handHistory = this.gameState.action_history || [];
            }
            
            // Call API to analyze hand
            const response = await fetch('/api/coach/analyze-hand', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    hand_history: handHistory,
                    game_state: this.gameState
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }
            
            const analysis = await response.json();
            
            // Store analysis data for later use
            window.currentAnalysisData = analysis;
            
            // Display analysis modal
            if (window.handAnalysisModal) {
                window.handAnalysisModal.show(analysis);
            }
        } catch (error) {
            // Show user-friendly error message
            let errorMsg = 'Analysis unavailable, please try again';
            if (error.message) {
                if (error.message.includes('timeout') || error.message.includes('time')) {
                    errorMsg = 'Analysis is taking longer than expected. Please wait...';
                } else if (error.message.includes('network') || error.message.includes('fetch')) {
                    errorMsg = 'Connection issue. Please check your network and try again.';
                } else if (error.message.includes('rate limit') || error.message.includes('rate')) {
                    errorMsg = 'Too many requests. Please wait a moment.';
                } else {
                    errorMsg = error.message;
                }
            }
            this.updateStatus(errorMsg);
            
            // Show error notification
            this.showNotification(errorMsg, 'error');
        }
    }

    startPolling() {
        this.stopPolling();
        this.pollInterval = setInterval(async () => {
            // Reset isProcessing if it's been stuck for too long (safety mechanism)
            if (this.isProcessing) {
                // If we're waiting for action and it's our turn, reset processing flag
                // This handles cases where an action request might have failed silently
                if (this.gameState && this.gameState.is_waiting_for_action && 
                    this.gameState.current_player === 0) {
                    this.isProcessing = false;
                } else {
                    return; // Still processing, skip this poll
                }
            }

            try {
                const response = await fetch(`/api/game/state?session_id=${this.sessionId}`);
                if (response.ok) {
                    const state = await response.json();
                    if (state && !state.error) {
                        // Check if it's AI's turn first (before checking for state changes)
                        // This ensures we catch the AI turn even if state hasn't "changed" yet
                        if (state.current_player === 1 && !state.is_over && !this.isProcessing) {
                            // Check if stage changed (new street dealt)
                            const previousStage = this.lastStage;
                            const currentStage = state.stage || 0;
                            const stageChanged = (previousStage !== null && previousStage !== currentStage);

                            // It's AI's turn - trigger AI action automatically
                            // This handles cases like after the flop when BB acts first, or preflop
                            // If stage changed (new cards dealt), wait for card animations before AI acts
                            if (stageChanged) {
                                setTimeout(() => {
                                    if (!this.isProcessing && state.current_player === 1 && !state.is_over) {
                                        this.processAITurn();
                                    }
                                }, 1200); // Wait for card animations (1000ms delay + 200ms buffer)
                            } else {
                                this.processAITurn();
                            }
                            return; // Exit early to avoid duplicate state updates
                        }
                        
                        // Only update if state changed significantly
                        if (!this.gameState ||
                            state.is_over !== this.gameState.is_over ||
                            state.current_player !== this.gameState.current_player ||
                            state.is_waiting_for_action !== this.gameState.is_waiting_for_action ||
                            JSON.stringify(state.legal_actions) !== JSON.stringify(this.gameState.legal_actions)) {

                            // Check if stage changed (new street dealt)
                            const previousStage = this.lastStage;
                            const currentStage = state.stage || 0;
                            const stageChanged = (previousStage !== null && previousStage !== currentStage);

                            this.gameState = state;
                            this.updateDisplay();

                            if (state.is_over) {
                                this.handleGameEnd();
                            } else if (state.current_player === 1 && !state.is_over && !this.isProcessing) {
                                // It's AI's turn - trigger AI action automatically
                                // This handles cases like after the flop when BB acts first
                                // If stage changed (new cards dealt), wait for card animations before AI acts
                                if (stageChanged) {
                                    setTimeout(() => {
                                        if (!this.isProcessing && this.gameState.current_player === 1 && !this.gameState.is_over) {
                                            this.processAITurn();
                                        }
                                    }, 1200); // Wait for card animations (1000ms delay + 200ms buffer)
                                } else {
                                    this.processAITurn();
                                }
                            }
                        }
                    }
                }
            } catch (error) {
            }
        }, 1000);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    showModal(modalId) {
        document.getElementById(modalId).classList.add('show');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('show');
    }

    showHelp() {
        this.showModal('helpModal');
    }
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.pokerGame = new PokerGame();
});

