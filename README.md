## Streaming Analytics Dashboard

- An end-to-end data analytics platform analyzing U.S. broadcast and streaming television viewership trends from 2019 to 2025, with machine learning-powered ratings forecasting and live survey integration.

- Live demo: streaming-analytics-dashboard.vercel.app

## Why Vercel? Have you heard the news?

- Quick and easy hosting.
- Project *could* work locally but having it public gives it much more freedom.
- Helps gather data with SurveyMonkey api integration.

## Features

- Platform Analytics: tracks the structural shift in U.S. TV consumption across broadcast, cable, and streaming from 2019 to 2025. Key events (COVID lockdown, Tokyo 2021 Olympics, Paris 2024 Olympics) are annotated directly on the charts. Peacock subscriber growth is plotted against Comcast quarterly earnings data.

- Graphs & Stats: four tabs of deep analytics: platform share trends with 12-week rolling averages, year-over-year change by platform, genre performance across all six seasons, and a linear regression forecast with 95% confidence intervals projectable up to 156 weeks out.

- Show Explorer: keyword search across 83 real NBC show-season records. Type a show name, genre, timeslot, or any keyword to filter. Results render as multi-season viewership trend charts, 18-49 demo rating trends, and a sortable data table with peak vs latest comparison.

- Predictive Model: a real scikit-learn Gradient Boosting Regressor (400 estimators, trained on real Nielsen data) served via FastAPI. Enter genre, network, timeslot, live flag, budget tier, tentpole event, reboot status, critic score, and season number to get a base prediction, adjusted forecast, and conservative/optimistic range. A benchmark chart shows how the prediction compares to similar shows from the same genre.

- What-If Simulator: pick any NBC or Peacock show and apply a content strategy scenario (move to streaming, add live elements, simulcast, reboot, prime time slot, limited series) to project viewership impact through 2030. All six scenarios are compared in a table.

- Retention & Churn: subscriber lifecycle modeling with four tabs: cohort retention curves by subscriber type, an interactive churn simulator with sliders for churn rate and acquisition, behavioral engagement segmentation (Power Viewers, Casual Browsers, Event Subscribers, At-Risk), and a survey insights tab wired to live SurveyMonkey data.

- Chatbot Navigation: simple chat-bot feature for navigation of tools and discussion, able to download transcripts for later reference.

## Limitations

- There's only so much data (public, authentic sources) that can be integrated into this project, which may push to data skewing

- The chatbot only has Groq as it's LLM api key integration. I did this to raise awareness to the fact that Gemini has been recently restricting users/accounts of their free api key tiers, so you'll be able to get decent responses out of Groq

## Disclaimer

This is an independent analytical portfolio project. It is not affiliated with, endorsed by, or associated with NBCUniversal, Peacock, or any of their subsidiaries. NBC, Peacock, and related marks are trademarks of NBCUniversal Media, LLC. Viewership data is sourced from publicly available Nielsen reports via TVSeriesFinale.com.

## Coming soon:

- Mapped out system design in the docs.

- More updated data, either integrated manually overtime or updated automatically via API. 
  
