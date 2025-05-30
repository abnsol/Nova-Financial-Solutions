# Nova-Financial-Solutions

## Project Overview

This project focuses on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements[cite: 2]. The primary objective is to enhance Nova Financial Solutions' predictive analytics capabilities by leveraging news sentiment to forecast stock price fluctuations and inform investment strategies[cite: 7, 15, 17]. This challenge refines skills in Data Engineering (DE), Financial Analytics (FA), and Machine Learning Engineering (MLE)[cite: 3].

## Business Objectives

The analysis has a two-fold focus:
1.  **Sentiment Analysis:** Quantify the tone and sentiment of financial news headlines using NLP techniques, associating scores with stock symbols[cite: 10, 11].
2.  **Correlation Analysis:** Establish statistical correlations between news sentiment and corresponding stock price movements, considering publication dates[cite: 12, 13, 14].

## Dataset

The Financial News and Stock Price Integration Dataset (FNSPID) combines quantitative and qualitative data[cite: 18]. Key fields include:
* `headline`: Article title [cite: 19]
* `url`: Link to article [cite: 20]
* `publisher`: Author/creator [cite: 21]
* `date`: Publication date and time (UTC-4) [cite: 21]
* `stock`: Stock ticker symbol [cite: 22]

## Interim Submission Progress

This interim submission covers the completion of Task 1 and initial data preparation for Task 2.

### Task 1: Environment, Git & Exploratory Data Analysis (EDA)

* **Environment & Git:** Established a reproducible Python data science environment and configured Git version control with a clear repository structure[cite: 23, 29].
* **EDA:** Performed comprehensive Exploratory Data Analysis on the financial news dataset[cite: 24].
    * Obtained descriptive statistics for headline lengths[cite: 32].
    * Counted articles per publisher to identify active sources[cite: 33].
    * Analyzed publication dates for trends and frequency variations[cite: 34, 36].
    * Conducted preliminary text analysis to identify common keywords and events[cite: 35].
    * Examined publishing times for peak news release periods[cite: 38].
    * Investigated publisher contributions and news types[cite: 39, 40, 41].

### Task 2 (Partial): Quantitative Analysis Preparation

* **Data Loading:** Loaded stock price data into a pandas DataFrame, ensuring required 'Open', 'High', 'Low', 'Close', and 'Volume' columns are present for technical indicator calculations[cite: 43, 44].

## Next Steps

Future work will focus on completing Task 2 and executing Task 3:

* **Complete Task 2:** Apply technical indicators (MA, RSI, MACD) using TA-Lib and PyNance, and visualize their impact on stock prices[cite: 25, 45].
* **Execute Task 3:** Align news and stock data by date[cite: 48, 54]. Perform sentiment analysis on headlines[cite: 49, 55]. Calculate daily stock returns[cite: 50, 56]. Conduct correlation analysis between sentiment and returns[cite: 51, 57, 58].
