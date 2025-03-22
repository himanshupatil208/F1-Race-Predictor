Title: F1 Race Winner Predictor 

Problem Statement:
The project’s objective is to accurately predict the Formula 1 (F1) Race Winner using machine learning models leveraging the data from past years grand prix’s such as race winners, race details, weather, teams data in order to assist team stakeholders, team managers make informed decisions.

Background:
Formula 1 (F1) is the highest class of international single-seater auto racing, regulated by the Fédération Internationale de l'Automobile (FIA). It conducts a series of races called grand prix on various circuits and streets all around the world. Each F1 team, often referred to as a "constructor," designs and manufactures its own cars. Teams typically consist of two drivers who compete for both individual and team championships over the season.
Formula 1 cars are built and tested according to FIA regulations for every season. These cars are driven at high speeds of 330 km/hr to 350 km/hr which are affected by smallest of factors such as tyre, starting grid, 

Impact
There is high investment in this sport. According to FIA regulation, a team can spend up to 135 millions in manufacturing the car. So, the teams and their stakeholders want to maximize their chances to win the championship. Our project will help to analyze past performance to understand what can be done better and predict the chances of the race winner. 

















DataSets: 

https://www.kaggle.com/datasets/debashish311601/formula-1-official-data-19502022
The dataset includes all race, sprint, qualifying, practice, driver, team, lap data among others for all the championships from 1950 until today.
The following files are included in this dataset:
race_details.csv: Includes detailed race results for all Grand Prix tracks from 1950 until present.
race_summaries.csv: Includes summarized race results for all Grand Prix tracks from 1950 until present.
starting_grids.csv: Includes the starting grids for all Grand Prix tracks from 1950 until present.
sprint_grid.csv: Includes the starting grid for the sprint race on a Saturday. The sprint format was introduced in 2021 for three tracks (Silverstone, Monza and Interlagos). The result of the sprint race decides the starting grid for the main race on Sunday.
sprint_results.csv: Includes sprint race results all for all Grand Prix tracks from 2021 until present.
fastest_laps.csv: Includes fastest lap summaries for all Grand Prix tracks from 1950 until present.
fastestlaps_detailed.csv: Includes fastest lap details for all Grand Prix tracks from 1950 until present.
qualifyings.csv: Includes detailed race results for all Grand Prix tracks from 1950 until present.
practices.csv: Includes FP1/FP2/FP3 results for all Grand Prix tracks from 1986 until present.
pitstops.csv: Includes pitstop details for all Grand Prix tracks from 1994 until present.
team_details.csv: Includes team details for teams that drove from 1958 until present.
constructor_standings.csv: Includes team standings for all Grand Prix tracks from 1958 until present. Team awards have been awarded only since 1958.
driver_standings.csv: Includes driver standings for all Grand Prix tracks from 1950 until present.
driver_details.csv: Includes driver details for all races from 1950 until present.





Data Cleaning/Processing:


Combining Dataset:
There are 14 files in the dataset from Kaggle that we combined using inner joins based on multiple columns [driver, grand prix and year] to form a dataset we can use to perform EDA on. 

Removed all data about grand prix prior to 2010.
For our  analysis we found that data past 2010 is irrelevant as the drivers, cars, teams are retired, changed and won’t have any positive impact on our prediction. We also dropped a number of files in the dataset from consideration such as practices.csv, fastest_laps.csv, sprint_results, sprint_grid, fastestlaps_detailed, constructor_standings and Qualifyings. The data in these files could not play an impactful role in the analysis resulting in 10554 rows in the dataset.

Remove Duplicates: 9814
There were around 742 duplicate rows, so we removed all this data from the dataset resulting in 9814 rows in our final dataset.

Feature Normalization: 
In the dataset, if the driver could not finish the race, his position is assigned DNF or NC. Since this value  cannot be analyzed, we changed these values to integers so we can use them in our analysis. DNF to 21 and NC to 0. The ‘Position’ and ‘Time’ fields were in string format so we converted them to integers and Date and Time format. 
Over the years teams have changed names after sponsors have joined or left the teams. This resulted in different names for teams in the dataset even though they are the same team. We combined these teams to the first mentioned team name. Eg: 'Force India Sahara': 'Force India Mercedes',   'Racing Point Bwt Mercedes',  

Null value handling: 
Since 2011, many driver’s have left F1 and new drivers have joined. The data for these drivers was Null in certain columns such as time for lap. As the ‘Time’ field won’t play a major role in race winner prediction, we dropped the column completely.






Exploratory Data Analysis (EDA):

1. Understanding the data:
We use the describe function on our data which gives statistical analysis.



Figure 1
Here Stops indicate the number of pit stops. Generally the number of pit stops is 1, 2 max 3. In our case the max value of Stops is 6 which indicates there is presence of outliers.




2.  Correlation of Features:
We use correlation matrix to understand relation between variables and how they affect the final position.
We use the numerical values like Pos_grid, Stops, No_pitstop.

  
Figure 2



Observation:
Pitstop vs Race Position: The correlation coefficient between the number of pit stops and race position is 0.011, which indicates a weak positive correlation. This suggests that, generally, racers who make more pit stops tend to finish higher in the race. However, the relationship is not very strong, and other factors likely play a more significant role.

Starting Position vs  Finishing Position: Weak Positive Correlation with Race Position: The correlation coefficient between starting position and race position is 0.33, which indicates a weak positive correlation. This suggests that, generally, racers who start further up on the grid tend to finish higher(lower position indicates winner and higher position indicates last) in the race. However, this relationship is not very strong, and other factors like driving skill, car performance, and race strategy also play a significant role.






















3. Number of Wins by Drivers over the years:

We wanted to observe the drivers performance over the years so we plotted a bar graph.



Figure 3

Observation:

Over the years, it can be observed that a certain driver dominates that particular year. There is a huge gap between the champion driver’s number of wins vs other driver’s number of wins.













4. Ratio of Number of races driver participated vs number of wins:

To see the relation between the longevity of driver in the F1 vs number of wins. 



Figure 4

Observation:

It can be observed that there is not a significant relation between the two. Drivers such as Lewis Hamilton seem to be an exception, but most of the drivers have low wins compared to the number of races they participated in.

So, we can’t simply use a linear  regression in determining the race winner based on past number of participation.






5.  Driver Wins Vs Location:

We wanted to see if there is any relation between the location and driver’s, if certain driver’s dominate in a particular location.




Figure 5

	Observation:

It seems there is a small relation between the location between the driver. Lewis Hamilton has a high winning streak in Britain and Spain, whereas Sebastian Vettal has a high number of wins in Singapore. Most of the drivers have either 2 or 3 wins at a particular location which can play a role in predicting a winner at a particular location.


	





6. Starting Position vs Winner

To observe the relation between if the starting 1st will result in a win.



Figure 6

	Observation:

It can be observed that there is a very strong relation between if someone starts first, they win the grand prix. 













7. Identifying outliers



Figure 7


Observation:
The number of pit stops are generally 1-2. The Maximum number of pit stops that we can have is 3. This was also confirmed when we first described the data.
The box plot above clearly shows the presence of outliers 4, 5 and 6.


In phase 1, we cleaned the data and performed EDA to explore, understand and analyse the data. In phase 1, we found that Starting Position of the driver plays a crucial role in predicting the winner. We also observed that one particular driver generally dominates in a season(year) with wins. In phase we are using this information and applying classification algorithms to predict the winner of the race.

Why GridSearchCV ?

It systematically tests the combinations of the hyperparameters values to identify the best performing model. 
It automates the tunning, uses k fold cross validation to assess the model performance on training data reducing the risk of overfitting.
Optimizes for a specific metric to focus on the desired performance.

Algorithms:

Baseline: A dummy classifier that randomly decides a winner has an accuracy of 1/25 = 0.04 = 4%

We decide a new baseline using a weaker model, and then use much more complex models to fit our data and capture non-linear relations. 
The baseline is the one against which we compare our results to justify the new model’s performance and whether investing in deploying the model is worthwhile.














Logistic Regression
We first load the data from the csv file and sort it by date.
We select Driver POS Standings, Drive Race Points, Laps, Team Race Points, Team Champ Points, Driver Champ Points and Driver Champ Positions as the features. We kept last 2 years of data for testing and used the remaining data for training. Means we will use the previous year data and predict the winners of 2021 and 2022. We scale the features to make sure all features are on a similar scale.
In the end we print the classification report and winners for 2021 and 2022 sorted according to race number.


As we can see the model has high accuracy, but the recall is 0.61 means the model performs good at identifying actual winners. Also, in the predictions we can see that for certain races we have multiple winners this is because these racers have probability above the threshold.
So, we do hyperparameter tuning to check if we can achieve a better recall.
After hyperparameter tuning the recall has improved from 0.61 to 0.77 meaning that our model has improved at identifying winners.

Usually in a race there’s just a single winner per race so we have an imbalance in data as there are nearly 20-25 losers per race and just a single winner. The improvement in recall without affecting accuracy indicates that the model has become better at handling imbalance.





As you can see the model is good at predicting true positives and true negatives.
 
In this we can see that drive race points have the highest positive coefficient value meaning that the driver’s points contribute significantly to winning. Which is true as the driver with the highest points wins.

The ROC curve is close to the top-left corner indicating excellent performance.
 
 
The AP value of 0.68 indicates a reasonably good performance.










KNN
The KNN model was chosen because its characteristics align well with the requirements and nature of data. As in KNN there are no assumptions about data distribution, as it can take a dataset of any size and can handle high dimensionality data.
We start with loading the dataset (initial_cleaned_winner_data.csv).
Target variable: Winner

Above are the available features out of which we have used only the important features using the correlation analysis.

Correlation Analysis:
Here we have computed the correlation matrix to get the most relevant features to target “Winner” which will train the model to get more accurate predictions.
selected_features = correlation_target[correlation_target > 0.4].index.tolist()

We have set the threshold here to 0.4, as it belongs to the moderate threshold range which calculates moderately correlated features. 
When we use a low threshold there is risk of using irrelevant features which may lead to more wrong predictions.
And when we use a high threshold we may exclude potential but weakly correlated features.

KNN relies on distance calculations, so irrelevant or weak features can affect the effect of meaningful features. A moderate threshold will be useful to get relevant predictors removing noises.

Below is the bar graph showing the importance of each feature on correlation with the target Winner.


So from above we get,
Final Selected Features
['Driver POS Standings', 'Team Race Points', 'Team Champ Wins', 'Driver Champ Points', 'Driver Champ Pos', 'Driver Champ Wins', 'Winner']

Later on we split the data into training and test set.
Training set : 80%
Test set : 20%

Hyperparameter Tuning:
Before training the model we are doing hyperparameter tuning, as it maximizes the model performance by systematically testing parameters combinations.

In the code we have used GridSearchCV to tune the hyperparameters. 
 	Below are the Key hyperparameters tuned:
	n_neighbors: The number of neighbors considered when making predictions
weights: We have “weights = distance ”, which indicates that closer neighbors have  more influence and “weights = uniform” when all neighbors contribute equally.
metric: Distance metric for calculating the similarity.
	E.g. Euclidean, Manhattan, Minkowski


	Later we get best parameters,
	Best Parameters: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}
	
Using the above parameters we do the predictions and then calculate recall and  accuracy of the mode.

Before hyperparameter tuning we get below results:

And after hyperparameter tuning we get,


Hence after parameter tuning recall has improved from 0.5 to 0.53 meaning that our model has improved at identifying winners.

Then we plot the confusion matrix. The confusion matrix visualizes true positives, false positives, false negatives, and true negatives.






The ROC curve plot evaluates the performance of a K-Nearest Neighbors (KNN) classifier. It shows the relation between the true positive rate and false positive rate.
The area under the curve 0.89 which indicates the strong ability of the model to distinguish between the classes.



	
	




















XGBoost:
XGBoost was chosen because it is a boosting model which is considered to be one of the most powerful models.
It provides parallel boosting and is the leading machine learning library/algorithm for most regression, classification problems.

Using a simple XGBoost model (with default values) gives us the following results:

A recall of 0.72 which is better than most algorithms (even after hyperparameter tuning).

Optimizing hyperparameters:
n_estimators: The number of trees (boosting rounds) in the model; higher values increase complexity but may risk overfitting.
max_depth: The maximum depth of each tree
learning_rate: The step size at each iteration to reduce errors; smaller values make training slower but more precise.
subsample: The fraction of the training data used for each boosting round; prevents overfitting by introducing randomness.
colsample_bytree: The fraction of features randomly sampled for each tree; controls model diversity and reduces overfitting.


	
	Running GridSearch on these variables gives us the following result:
		
i.e. a Recall of 0.78

With these hyperparameters:
'colsample_bytree': 1.0
'learning_rate': 0.05
'max_depth': 3
'n_estimators': 300
'subsample': 1.0

The confusion matrix of the best model can be seen below:

with minimum number of misclassifications

The feature importance graph shows that ‘Driver Champ Wins’ is the most important feature:


whereas ‘Team Country’ is the least

The Precision-Recall Curve shows that the model is a good fit (with AUC = 0.85):

We also plot a n_estimators vs Recall score graph to verify that more estimators increases the recall needed:



Neural Networks
	Class 0 -> Not Winner
	Class 1 -> Winner
Reason for Selection
We have selected this algorithm to train our model due to their ability to model complex, non-linear relationships in the data.
They are suitable for high-dimensional datasets and can automatically learn relevant patterns, making them ideal for the Winner prediction task.
Baseline Accuracy/ Recall
Using a simple model with 2 hidden layers we get a recall of 54.39%
		
Adam optimizer
The Adam optimizer was selected due to its ability to adjust the learning rate which enables the model to converge quicker and demonstrate good performance on high dimensional data.
Hyperparameter Tuning
Since the previous network is a simple 2-hidden layer network, we add more hidden layers to enable it to capture deeper connections.
Best Accuracy
Using 5 hidden layers allows us to increase the recall from 54% to 69% for the same number of iterations. However the recall plateaus there and doesnt seem to increase any more.
		
Final findings
Even though the accuracy and recall are far better than our original baselines, a neural network with BCELoss is not the best model (even with deep connections) for this problems



























Random Forest:
Next model we used is the Random Forest Classification model. We chose a random forest model because it works by combining the predictions of multiple decision trees to make more stable and accurate predictions. It is also useful as it avoids overfitting and provides Feature Importance.

We selected ‘season','Driver Code','Team Code', 'Driver POS Standings', 'Driver Champ Points', 'Team Champ Points' features to train the data.

We also converted the ‘Winner’ Column (target) to binary format (0 or 1) to declutter the target field and simplify the prediction.

While splitting data, data was stratified according to ‘seasons’.
 	      

To find the best hyperparameters we use RandomSearchCV() method and set the cross validation to 5, scoring metric as recall and number of iteration to 200. 


We retrieved the best params from the hyperparameters and trained the RandomForestClassifier with them and set class_weight=balanced to control the imbalance in the dataset. Since there is only 1 winner per race and 19-24 losers, the ratio is highly unbalanced. 

After performing the prediction, to convert it into binary, we set a threshold of 0.5. If the prediction is above 0.5, we set it to 1. (shown in above screenshot)

Below is the result - Best Parameters, Recall and Accuracy:
Recall: 0.36
Accuracy: 0.95	


Even though the model’s accuracy is high, the recall is 0.36 which is poor. Thus, the hyper parameters require tuning.

We figured out that the reason for the model poor performance in True Recall is that the data has been stratified by season. 


After tuning the data split by not shuffling it and hyper parameters, the output of the code is below:
Recall: 0.69
	Accuracy: 0.93



Confusion Matrix;

There are 63 correct winners predicted by the model.

The feature importance graph shows that ‘Driver POS Standing’ i.e Drivers starting grid position’ is the most important feature followed by ‘Driver’s Championship Points’ 
	




Findings:
Best Model: XGBoost with .78 recall.
Worst Model: KNN with .53 recall.

