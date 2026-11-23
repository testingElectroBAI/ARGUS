# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

ARGUS is for the developer's personal AI research and analysis.

## Product Purpose

ARGUS provides a local browser interface for asking an AI assistant questions and analyzing image or video content.

## Capabilities and Constraints

- Supports text questions, image and video uploads, webcam snapshots, and screen captures.
- Routes requests through automatic, vision-only, or reasoning-only model modes.
- Can return web sources and runtime request statistics.
- Conversation context and captures live in browser memory; conversation state is lost on reload.
- The interface is served locally by the existing Flask application. UI implementation remains plain HTML, CSS, and JavaScript.

## Brand Commitments

Keep the ARGUS name. The redesign must remain minimal, clean, modern, and avoid unnecessary visual effects. The supplied 21st.dev background and AI chat input references guide the interface.

## Evidence on Hand

The existing Flask routes, request flow, and `templates/index.html`, `static/style.css`, and `static/script.js` are the product evidence. Do not fabricate product claims or persistent conversation storage.
