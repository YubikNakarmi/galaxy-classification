# Galaxy Zoo 2 Dataset
The [Galaxy Zoo 2](https://arxiv.org/abs/1308.3496) project by (Willett et al., 2013), which contains morphological classifications of galaxies based on volunteer votes. The votes are decided using tree based questiosn, where each question has morphogological features as answers.

 The desiciion tree has 11 main questions with 37 possible answers. Each answer is represented as a column in the dataset, and the value in each column represents the fraction of volunteers who voted for that answer for a given galaxy. The desicion tree is shown below:

![Figure 1: Galaxy Zoo 2 decision tree](../assets/galaxy_tree(willet.w).png){align="center" width="80%"}

/// caption
Figure 1: Galaxy Zoo 2 decision tree [Willett et al., 2013](https://arxiv.org/abs/1308.3496)
///

About 16 million classifications were made by volunteers for 304,122 galaxies, resulting in a rich dataset for morphological analysis. 

The dataset contains a total of 304,122 samples, each corresponding to a unique galaxy. The dataset is available in CSV format and can be accessed through the [Galaxy Zoo 2 data release](https://data.galaxyzoo.org/). Ultimately voting fractions for each answer are used as labels for training machine learning models to classify galaxy morphologies. 

# Data Splits
Data is split into training, validation, and test sets using an 80-10-10 split. The training set is used to train the model, the validation set is used for hyperparameter tuning and model selection, and the test set is used for final evaluation of the model's performance.
