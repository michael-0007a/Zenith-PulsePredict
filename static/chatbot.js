// Place this file in the static folder as "chatbot.js"
document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form submission
    const userMessage = document.getElementById('user_message').value;

    const response = await fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            user_message: userMessage
        })
    });

    const chatbotResponse = await response.text();

    const messageBox = document.getElementById('message-box');
    messageBox.innerHTML += `<div class="user-message">${userMessage}</div>`;
    messageBox.innerHTML += `<div class="chatbot-response">${chatbotResponse}</div>`;

    // Clear input field and scroll to the latest message
    document.getElementById('user_message').value = '';
    messageBox.scrollTop = messageBox.scrollHeight;
});
