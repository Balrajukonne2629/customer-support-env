
---
title: Customer Support Triage
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# Customer Support Triage OpenEnv

An OpenEnv environment for AI agents to manage customer support tickets.

## Tasks

| ID | Name | Difficulty | Tickets |
|----|------|-----------|---------|
| task1 | Ticket Classification | Easy | 7 |
| task2 | Priority and Routing | Medium | 5 |
| task3 | Full Ticket Resolution | Hard | 3 |

## API

- POST /reset - start episode
- POST /step - take action
- GET /state - current state
- GET /health - health check

## Baseline Scores

| Task | Score |
|------|-------|
| task1 | 0.71 |
| task2 | 0.68 |
| task3 | 0.52 |