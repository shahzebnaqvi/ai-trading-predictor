// content.js
// No changes needed for basic functionality
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'getSymbol') {
        const url = window.location.href;
        const symbolMatch = url.match(/symbol=([^&]+)/) || url.match(/symbols\/([^\/]+)/);
        sendResponse({
            symbol: symbolMatch ? symbolMatch[1] : null
        });
    }
    return true;
});