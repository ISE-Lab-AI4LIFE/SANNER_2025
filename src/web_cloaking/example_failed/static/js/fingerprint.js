fetch("/fp", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        width: window.screen.width,
        height: window.screen.height,
        colorDepth: window.screen.colorDepth,
        webdriver: navigator.webdriver || false
    })
});
