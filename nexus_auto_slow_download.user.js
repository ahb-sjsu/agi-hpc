// ==UserScript==
// @name         Nexus Auto Download
// @namespace    storywealth
// @version      11.0
// @description  Auto-clicks Slow download inside Nexus shadow DOM components
// @match        https://www.nexusmods.com/*/mods/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    if (!window.location.search.includes('file_id=')) return;

    // Wait 8 seconds for page animations and shadow DOM to render
    setTimeout(function() {
        var n = 0;
        var iv = setInterval(function() {

            // Search inside shadow DOM of custom elements
            var components = document.querySelectorAll('mod-file-download, slow-download-prompt');
            for (var i = 0; i < components.length; i++) {
                var shadow = components[i].shadowRoot;
                if (!shadow) continue;

                // Look for any clickable with "Slow download" text
                var els = shadow.querySelectorAll('button, a, span, div');
                for (var j = 0; j < els.length; j++) {
                    var t = els[j].textContent.trim();
                    if (t === 'Slow download' || t === 'Slow Download') {
                        var btn = els[j].closest('button') || els[j].closest('a') || els[j];
                        btn.click();
                        clearInterval(iv);
                        return;
                    }
                }
            }

            // Also try dispatching the slowDownload event directly
            var modComponent = document.querySelector('mod-file-download');
            if (modComponent && n === 5) {
                modComponent.dispatchEvent(new Event('slowDownload'));
            }

            if (++n > 30) clearInterval(iv);
        }, 1000);
    }, 8000);
})();
