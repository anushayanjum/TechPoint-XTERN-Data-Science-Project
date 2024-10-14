# VeloCityX Data Analysis and Fan Challenge Proposal

## Overview

VeloCityX is an innovative application designed to enhance the spectator experience in the world of autonomous racing. The app features immersive 360° live coverage, interactive fan challenges, virtual merchandise purchases, and sponsorship integration. This project involves using data science to analyze user interactions with the app and provide insights that drive fan engagement and monetization strategies.

As part of this analysis, we:

- Cleaned and prepared the provided dataset for analysis.
- Conducted clustering and predictive modeling to uncover key insights into user behavior.
- Proposed an enhanced fan challenge aimed at boosting engagement and monetization.

## Project Structure

The project consists of the following main components:

1. **Data Collection and Inspection**  
   Loaded and inspected the dataset to understand its structure and key attributes. This included handling missing data and converting data types where necessary.

2. **Data Cleaning**  
   Processed the dataset by handling missing values, converting data types, and removing outliers. This ensures the dataset is in a usable format for analysis.

3. **Clustering Analysis and PCA**  
   Applied clustering techniques to segment users based on their interactions. Principal Component Analysis (PCA) was used to reduce the dataset's dimensionality for better interpretability.

4. **Merchandise Purchase Trends**  
   Analyzed trends in virtual merchandise purchases to identify key factors that influence user purchasing decisions.

5. **Predictive Modeling**  
   Built classification models (Logistic Regression, Decision Tree, Random Forest) to predict user likelihood of purchasing virtual merchandise based on their engagement with the app.

6. **Proposed Enhanced Fan Challenge**  
   Developed a multi-stage predictive challenge that allows fans to engage throughout races by making predictions on race outcomes, pit stops, and team performance. This challenge is designed to increase engagement and boost monetization.

## Dataset

The dataset includes the following features:

- **User ID**: Unique identifier for each user.
- **Fan Challenges Completed**: Number of challenges completed by each user.
- **Predictive Accuracy**: Percentage of correct predictions made in fan challenges.
- **Virtual Merchandise Purchases**: Number of virtual merchandise items purchased by each user.
- **Sponsorship Interactions**: Number of ad clicks during sponsorship integrations.
- **Time Spent on "Live 360" Coverage**: Time in minutes that users spent on live race coverage.
- **Real-Time Chat Activity**: Number of messages sent by users during live races.

## Key Findings

- **Clustering**: Identified distinct clusters of users with varying engagement levels. Some users were highly active in chats but less engaged with merchandise, while others had high sponsorship interaction and merchandise purchases.
  
- **Merchandise Trends**: Users who spent more time on "Live 360" coverage and interacted with sponsorships were more likely to purchase virtual merchandise.

- **Predictive Modeling**: Random Forest models were the most effective in predicting whether a user would purchase virtual merchandise, though model performance could be improved with further tuning.

## Enhanced Fan Challenge Proposal

### Multi-Stage Predictive Challenge
A new challenge designed to keep fans engaged throughout the race by making multiple predictions on:

1. Which vehicle will run the longest before a pit stop.
2. The number of pit stops a team will take.
3. Bonus predictions on the fastest recharge times.

### Engagement and Monetization Strategies

- **Dynamic Quests**: Real-time quest chains that reward users for accurate predictions.
- **Team-Based Challenges**: Fans can align with their favorite racing teams to earn team-specific rewards.
- **Exclusive Merchandise**: Fans can unlock team-specific virtual merchandise by completing challenges.
- **Prediction Power-Ups**: In-app purchases that give users additional prediction chances or hints.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/velocityx-data-analysis.git
