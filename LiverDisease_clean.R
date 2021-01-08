
# load required packages and if not installed, install them

list.of.packages <- c("tidyverse", "caret", "data.table", "stringr", "lubridate")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(tidyverse)
library(caret)
library(data.table)
# library(stringr)
# library(lubridate)
library(ggpubr)
library(gridExtra)
library(corrplot)
library(matrixStats)
library(moments)
library(purrr)
library(gam)
library(RColorBrewer)
library(ggrepel)
library(rpart.plot)
library(imbalance)



# Download data ----

liver <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian Liver Patient Dataset (ILPD).csv")

colnames(liver) <- c("Age", "Gender", "totalBilirubin", "directBilirubin", "totalProteins", "Albumin"
                     ,"AGratio", "SGPT", "SGOT", "Alkphos", "Disease")


# turning a "1" into "yes" and "2" into "no", because it is a bit difficult to follow a class variable defined as 1s and 2s
# and "yes" (disease) and "no" (no disease) are more easy to follow.

liver$Disease[liver$Disease == 1] = "yes"
liver$Disease[liver$Disease == 2] = "no"




# Introduction ----

# Patients with Liver disease have been continuously increasing because of excessive consumption of alcohol,
# inhale of harmful gases, intake of contaminated food, pickles and drugs.

# The data set was collected from north east of Andhra Pradesh, India.

# This dataset will be used to evaluate prediction algorithms in an effort to reduce burden on doctors.

# This report aims at using various classification machine learning techniques to predict if
# a patient has a liver disease or not.



## Data exploration ----


# Before doing any data exploration and pre-processing, the original liver dataset
# will be split into a "validation" set and a "liver subset", in order to mimic
# real-life machine learning techniques, where the dataset to be predicted is generally unknown.
# Therefor the "validation" dataset will be used as the final test set, on which the best
# performing algorithm will be used.

# The "liver subset" will then be further split into train and test sets, on which several
# machine learning methods will be applied.

# In both steps the datasets will be split into 20% for testing and 80% for training.
# The choice for using 20% and not less is due to the relatively small size of the overall dataset,
# which enables the test dataset to be large enough to perform validation of the different models.


set.seed(1, sample.kind="Rounding")

test_index   <- createDataPartition(y = liver$Disease, times = 1, p = 0.2, list = FALSE)
liver_subset <- liver[-test_index,]
validation   <- liver[test_index,]

rm(test_index)


# Check if the validation and liver_subset datasets have similar proportions of liver disease

mean(liver_subset$Disease == "yes")
mean(validation$Disease == "yes")

# it looks like the 2 datasets are balanced


# From now on all data exploration and pre-processing will be done on liver_subset.


# Check if there are any null values in the dataset and if yes, drop those rows:

liver_subset[rowSums(is.na(liver_subset))!=0,]

# Removing the rows with NA values from the dataset.

liver_subset <- na.omit(liver_subset)


# Check if there are duplicate rows in the dataset:

liver_subset[duplicated(liver_subset) == TRUE,]

# There seem to be 6 duplicated rows in the dataset, which might cause noise later on
# in the analysis. The duplicates will be removed from the dataset.

liver_subset <- unique(liver_subset)




# Create a secondary dataset (same data) but in a different structure, to allow for easier matrix calculations

# Convert the Disease (class label) column into factor and all other columns into a matrix column
tmp <- liver_subset %>% 
  mutate(Gender = as.numeric(Gender=="Male"))
x <- data.matrix(tmp[1:10])
Disease <- factor(liver_subset$Disease, levels = c("yes","no"))

# Merge the 2 sets back into the liver dataset as 2 lists
liver2_subset <- cbind(list(x),list(Disease))
names(liver2_subset) <- c("x", "y")

rm(tmp,x,Disease)




# Dataset information ----


# The overview of the dataset:
str(liver_subset)

# "Disease" is a class label used to divide the data into 2 groups (patients with a liver disease or not).

# The dataset also contains 10 features (or predictor variables): age, gender, total Bilirubin, 
# direct Bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos.
# The predictor variables will be used to determine if a patient has a liver disease or not.

# Any patient whose age exceeded 89 is listed as being of age "90" (as per the creators of the dataset).

# Attribute Information:
#   
# 1. Age: Age of the patient
# 2. Gender: Gender of the patient
# 3. totalBilirubin: Total Bilirubin
# 4. directBilirubin: Direct Bilirubin
# 5. Alkphos: Alkaline Phosphotase
# 6. Sgpt: Alamine Aminotransferase
# 7. Sgot: Aspartate Aminotransferase
# 8. totalProteins: Total Proteins
# 9. Albumin: Albumin
# 10. AGratio: Albumin and Globulin Ratio
# 11. Disease: Selector field used to split the data into two sets (labeled by the experts)



# This data set contains 
sum(liver$Disease == "yes")
# patients with a liver disease records and 
sum(liver$Disease == "no")
# patients with no liver disease records.

# This dataset contains
paste0(round(mean(liver$Disease == "yes") * 100), " %")
# of patients with a liver disease, which implies there is a
# class imbalance in the dataset. 
# I have found an article which mentions:
# "Nonalcoholic fatty liver disease (NAFLD) is emerging as an important cause of liver disease in India. 
# Epidemiological studies suggest prevalence of NAFLD in around 9% to 32% of general population in India 
# with higher prevalence in those with overweight or obesity and those with diabetes or prediabetes."
# (https://pubmed.ncbi.nlm.nih.gov/21191681/)

# Machine learning models on a dataset with high prevalence of disease might
# over-predict disease on the general population. This will be discussed in more detail
# in the Data preprocessing and Modelling sections.




# Analysis of the gender and age variables

p1 <- liver_subset %>%
  ggplot(aes(Gender, fill = Disease)) +
  geom_bar() +
  labs(title = "Gender", y = "Number of Patients")

p2 <- liver_subset %>%
  ggplot(aes(Age, fill = Disease)) +
  geom_bar() +
  labs(title = "Age", y = "Number of Patients") 

ggarrange(p1, p2, common.legend = TRUE)


# The gender graph reveals that the dataset contains many more males than females but
# having a liver disease seems to be prevalent for both genders 
# However males seem a bit more prevalent to have a liver disease than females (73.8% vs. 63.8%),
# but the difference is not significant. The number of females in the dataset is also quite smaller
# than males so the dataset is not balanced in this regard.

liver_subset %>%
  select(Gender, Age, Disease) %>% 
  group_by(Gender, Disease) %>% 
  summarise(count = n()) %>%
  group_by(Gender) %>% 
  mutate(share_per_gender = count / sum(count) * 100)


# Looking into age a bit closer reveals that having a liver disease happens at an average age of
# around 46, but the full range of liver patients goes from age 7 to age 90.

liver_subset %>%
  select(Gender, Age, Disease) %>% 
  group_by(Disease) %>% 
  summarise(avg = mean(Age), min = min(Age), max = max(Age), sd = sd(Age))




# The distribution of liver disease over age

p1 <- liver_subset %>% 
  filter(Gender == "Female") %>% 
  ggplot(aes(Age, color = Disease)) + 
  geom_density() +
  labs(title = "Disease distribution for Females") 

p2 <- liver_subset %>% 
  filter(Gender == "Male") %>% 
  ggplot(aes(Age, color = Disease)) + 
  geom_density() +
  labs(title = "Disease distribution for Males") 

p3 <- liver_subset %>% 
  ggplot(aes(Age, color = Disease)) + 
  geom_density() +
  labs(title = "Disease distribution - overall")

ggarrange(p1, p2, p3, common.legend = TRUE)


# Also viewed as a boxplot

liver_subset %>% 
  ggplot(aes(Age, Gender, fill = Disease)) +
  geom_boxplot()



# Based on the boxplot the age distribution of male diseased patients seems to be skewed more to the right
# than females (i.e. males that are a bit older than females seem to be getting a liver disease).
# The age plots also show us that relatively young people get a liver disease (somewhere between age 35 and 60),
# which could imply that having a liver disease is not an "old age" problem (at least in this small data sample).


# Some additional data analysis ----

# Computing box plots for continuous variables and jitter plots for categorical values:

b1 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,1]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[1])

b2 <- liver_subset %>%
  ggplot(aes(Gender, Disease, colour = Disease)) +
  geom_jitter(height = 0.2, width = 0.2) +
  labs(title = colnames(liver_subset)[2])

b3 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,3]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[3])

b4 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,4]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[4])

b5 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,5]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[5])

b6 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,6]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[6])

b7 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,7]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[7])

b8 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,8]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[8])

b9 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,9]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[9])

b10 <- data.frame(Disease = liver2_subset$y, liver2_subset$x[,10]) %>%
  gather(key = "Feature", value = "value", -Disease) %>%
  ggplot(aes(Feature, value, fill = Disease)) +
  geom_boxplot() +
  labs(title = colnames(liver2_subset$x)[10])

ggarrange(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, common.legend = TRUE)  


# Some initial conclusions can be drawn from these plots:
# Age, Gender, SGPT, SGOT and Alkphos seem to be relatively weak predictors,
# because the box plots/jitter does not seem to indicate the presence/or no presence of
# a liver disease.

# I will next perform several data pre-processing steps, which hopefully will shed more light
# into each predictor.





# Data pre-processing ----

# In machine learning, we often transform predictors before running machine learning algorithms. 
# We can also remove predictors if that are clearly not useful.

# Examples of pre-processing include standardizing the predictors, 
# taking the log transform of some predictors, removing predictors that are highly correlated with others, 
# and removing predictors with very few non-unique values or close to zero variation.




# Dataset resampling ----

# Due to the imbalanced nature of this dataset, I will attempt to oversample my overall training data.
# Inspiration for performing this step is taken from this article:
# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

# Oversampling is generally done on small datasets and is the process of adding instances from the 
# under-represented class to the training dataset.

set.seed(1, sample.kind="Rounding")

tmp <- liver_subset %>% 
  mutate(Disease = as.numeric(Disease == "yes")) %>% 
  mutate(Gender = as.numeric(Gender == "Male"))

liver_subset_sampled <- oversample(
  tmp,
  ratio = 0.7,
  method = "SMOTE",
  classAttr = "Disease"
) %>% 
  mutate(Gender = round(Gender))


# The new dataset  now contains
paste0(round(mean(liver_subset_sampled$Disease == 1) * 100), " %")
# of patients with a liver disease, which does result in a more balanced dataset.
# This is what will be used from now on in the analysis.


# Create a secondary dataset (same data) but in a different structure, to allow for easier matrix calculations

# Convert the Disease (class label) column into factor and all other columns into a matrix column

liver_subset_sampled$Disease <- ifelse(liver_subset_sampled$Disease == 1, "yes", "no")
tmp <- liver_subset_sampled
x <- data.matrix(tmp[1:10])
Disease <- factor(liver_subset_sampled$Disease, levels = c("yes","no"))

# Merge the 2 sets back into the liver dataset as 2 lists
liver2_subset <- cbind(list(x),list(Disease))
names(liver2_subset) <- c("x", "y")

rm(tmp,x,Disease)



# Data normalization ----

# As a first step in data processing, I will scale (or standardize) the data.
# The motivation for doing this step is that the features of the dataset have quite
# varying ranges (as we can see in the dataset overview).
# By standardizing the variables, we can make sure that we have comparable values in this analysis.

# This step is also important for producing more meaningful correlation matrix, principal component analysis
# and calculation of distances.

# In several iterations of standardization steps, I have used the R "scale" function,
# and the Min-Max normalization.

# I have decided to choose in the end the scale function, because this one
# produced the best predictions (even if only marginally so).



# Normalizing data with the scale function


liver2_subset_scale <- liver2_subset
liver2_subset_scale$x <- scale(liver2_subset$x)

# We can now see all feature values are in the range
min(liver2_subset_scale$x)
# to
max(liver2_subset_scale$x)

head(liver2_subset_scale$x)


# Checking for skewness in data ----

# Motivation: Skewed data is data that has a distribution that is pushed to one side or the other 
# (larger or smaller values) rather than being normally distributed. 
# Some machine learning methods assume normally distributed data and can perform better 
# if the skew is removed.


d_s <- as.data.frame(liver2_subset_scale$x)

ggplot(gather(d_s), aes(value)) + 
  geom_histogram(bins = 10) + 
  geom_density(aes(y=0.1*..count..), color = "red") +
  facet_wrap(~key, scales = 'free_x')

# "Skewness" factors:
s <- apply(d_s, 2, skewness, na.rm =TRUE)
s


# Although some of the predictors are skewed, further data transformations for removing skewness
# have proved to not improve the prediction power overall, and therefor is now left out of this report.



# Heatmaps

f1 <- dist(t(liver2_subset$x))
heatmap(as.matrix(f1), name = "Heatmap before data preprocessing")

# f2 <- dist(t(liver2_subset_norm$x))
# heatmap(as.matrix(f2), name = "Heatmap after data normalization")
# 
# f3 <- dist(t(liver2_subset_norm_tr$x))
# heatmap(as.matrix(f3), name = "Heatmap after norm. and transformation")

f4 <- dist(t(liver2_subset_scale$x))
heatmap(as.matrix(f4), name = "Heatmap after standardization")




# We can run the nearZero function from the caret package to see if features 
# do not vary much from observation to observation, as well check if there are any
# features with no variability

sds <- rowSds(liver2_subset$x)
qplot(sds, bins = 256) # there are no features with no variability

nzv <- nearZeroVar(liver2_subset$x) 
nzv
# the near zero variance function does not recommend the removal of any feature

sds <- rowSds(liver2_subset_scale$x)
qplot(sds, bins = 256) # there are no features with no variability

nzv <- nearZeroVar(liver2_subset_scale$x) 
nzv
# the near zero variance function does not recommend the removal of any feature




# Correlation Matrix of all features ----


c4 <- liver2_subset_scale$x %>%
  cor(use = "complete.obs")
corrplot(c4, tl.col = "black", addCoef.col = "black")




# From the correlation matrix, it looks like the following pairs of features are strongly correlated:
# total Bilirubin and direct Bilirubin
# AGratio (Albumin and Globulin ratio) and Albumin
# SGPT and SGOT
# Ikphos and SGOT

# Due to high correlation, some of these columns could be dropped.
# Motivation: some algorithms degrade in importance with the existence of highly correlated attributes.

# Before dropping any columns I will also have a look into the PCA analysis.



# Principal Component Analysis ----

# Principal Component Analysis, or PCA, 
# is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets,
# by transforming a large set of variables into a smaller one that still contains most of the
# information from the larger set.

# In this analysis I have don't have many dimensions, but I will use PCA
# to check if I can learn more about the training dataset.



pca <- prcomp(liver2_subset_scale$x)
summary(pca)


# Findings: around 80% of the variance is explained by the first 5 PCs

data.frame(type = liver2_subset_scale$y, pca$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()


# When looking at the boxplot, 
# we can see that the IQRs overlap for all PCs, but PC1 has the least overlaps.
# Generally, good predictive features should show distinctions in the boxplots (no or very
# little overlap).



# Studying various interpretations of the first 5 PCs, such as the examples below,
# suggests further that some of the highly correlated 
# variables can be dropped. Based on this interpretation, as well as running multiple 
# machine learning models with and without dropping variables,
# I have reached the conclusion that removing totalBilirubin, AGratio and SGPT can actually improve
# the predictive power of the final model.


data.frame(pca$x[,1:10], Disease=liver2_subset_scale$y) %>%
  ggplot(aes(PC1,PC5, fill = Disease))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)



pcs <- data.frame(pca$rotation, name = colnames(liver2_subset_scale$x))
pcs %>%  ggplot(aes(PC1, PC5)) + 
  geom_point() + 
  geom_text_repel(aes(PC1, PC5, label=name),
                  data = pcs)

# pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10)
# pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10)
# pcs %>% select(name, PC3) %>% arrange(desc(PC3)) %>% slice(1:10)
# pcs %>% select(name, PC4) %>% arrange(desc(PC4)) %>% slice(1:10)
pcs %>% select(name, PC5) %>% arrange(desc(PC5)) %>% slice(1:10)





# Classification Predictive Modeling ----

# A bit of introduction to the type of modeling I will apply in this report:
# Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).
# The output variables are often called labels or categories. The mapping function predicts the class or category for a given observation.
# For example, an email of text can be classified as belonging to one of two classes: “spam“ and “not spam“.
# 
# My dataset presents a two-class (binary) classification problem.

# There are many ways to estimate the skill of a classification predictive model, 
# but perhaps the most common is to calculate the classification accuracy.
# The classification accuracy is the percentage of correctly classified examples out of all predictions made.

# However on an imbalanced dataset such as this one with high prevalence of disease,
# we need to look into other measures as well.
# The other 2 measures would be sensitivity 
# (the proportion of liver patients who are correctly diagnosed) 
# and specificity (the proportion of non-liver patients that are correctly diagnosed).

# Another way of quantifying specificity is by the proportion of liver-diagnoses that are 
# actually liver-patients, also called precision.
# Precision in this analysis is also very important, since having high sensitivity in the model
# will not necessarily mean it is a good prediction, without also looking at precision.

# But it is also important to state that precision depends on prevalence, and our dataset has a prevalence
# of liver patients, which implies that I could get higher precision even when guessing.

# Furthermore, in the medical world, I would assume that it is most important to
# correctly diagnose liver patients (therefor having a high sensitivity), than correctly identifying the non-liver
# patients (therefor having a lower specificity should be fine).
# In other words, having a higher False Positive rate (healthy patients incorrectly diagnosed as diseased)
# is fine as long as the False Negatives is kept at its lowest (diseased patients incorrectly diagnosed as being healthy).
# My reasoning behind this is that is it far more dangerous to send people with a liver disease home without
# treatment than incorrectly identifying a liver disease in a healthy person, which can be eliminated
# later on by doing more tests etc.


# Further notes: A machine learning algorithm with very high sensitivity and specificity may not be useful 
# in practice when prevalence is close to either 0 or 1. 



## Split dataset into training and test datasets ----

# The dataset will be split into 20% for testing and 80% for training.

# Before splitting the dataset, 3 columns will be dropped (explained earlier).


# liver_final <- liver2_subset_norm_tr
# liver_final <- liver2_subset
liver_final <- liver2_subset_scale
# liver_final <- liver2_subset_norm


liver_final$x <- subset(liver_final$x, select = -c(totalBilirubin, AGratio, SGPT))


set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

test_index <- createDataPartition(liver_final$y, times = 1, p = 0.2, list = FALSE)
test_x <- liver_final$x[test_index,]
test_y <- liver_final$y[test_index]
train_x <- liver_final$x[-test_index,]
train_y <- liver_final$y[-test_index]

rm(test_index)


# Check if the training and test sets have similar proportions of liver disease
mean(train_y == "yes")
mean(test_y == "yes")
# it looks like the training and test sets are balanced


test <- data.frame(test_x, test_y) %>% 
  rename(Disease = test_y)
train <- data.frame(train_x, train_y) %>% 
  rename(Disease = train_y)




## Logistic regression model ----

set.seed(1, sample.kind="Rounding")

train_glm <- train(train_x, train_y, method = "glm")
pred_glm <- predict(train_glm, test_x, type = "raw")
cm <- confusionMatrix(pred_glm, test_y)


# Store the results in a dataframe
cm_results <- data_frame(method = "Logistic regression model (glm)",
                         Accuracy = cm$overall["Accuracy"],
                         Sensitivity = cm$byClass[c("Sensitivity")],
                         Specificity = cm$byClass[c("Specificity")],
                         Prevalence = cm$byClass[c("Prevalence")],
                         Kappa = cm$overall[c("Kappa")],
                         F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                         TruePositives = cm$table[1,1],
                         FalsePositives = cm$table[1,2],
                         FalseNegatives = cm$table[2,1],
                         TrueNegatives = cm$table[2,2])
cm_results

varImp(train_glm)



# Linear discriminant analysis ----

set.seed(1, sample.kind="Rounding")

train_lda <- train(train_x, train_y, method = "lda")
pred_lda <- predict(train_lda, test_x, type = "raw")
cm3 <- confusionMatrix(pred_lda, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Linear discriminant analysis (LDA)",
                                   Accuracy = cm3$overall["Accuracy"],
                                   Sensitivity = cm3$byClass[c("Sensitivity")],
                                   Specificity = cm3$byClass[c("Specificity")],
                                   Prevalence = cm3$byClass[c("Prevalence")],
                                   Kappa = cm3$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm3$table[1,1],
                                   FalsePositives = cm3$table[1,2],
                                   FalseNegatives = cm3$table[2,1],
                                   TrueNegatives = cm3$table[2,2]))
cm_results

varImp(train_lda)



# Quadratic discriminant analysis ----

set.seed(1, sample.kind="Rounding")

train_qda <- train(train_x, train_y, method = "qda")
pred_qda <- predict(train_qda, test_x, type = "raw")
cm4 <- confusionMatrix(pred_qda, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Quadratic discriminant analysis (QDA)",
                                   Accuracy = cm4$overall["Accuracy"],
                                   Sensitivity = cm4$byClass[c("Sensitivity")],
                                   Specificity = cm4$byClass[c("Specificity")],
                                   Prevalence = cm4$byClass[c("Prevalence")],
                                   Kappa = cm4$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm4$table[1,1],
                                   FalsePositives = cm4$table[1,2],
                                   FalseNegatives = cm4$table[2,1],
                                   TrueNegatives = cm4$table[2,2]))
cm_results

varImp(train_qda)



# k-nearest neighbors ----

# First run the model with no tuning:

set.seed(1, sample.kind="Rounding")

train_knn <- train(Disease ~ ., method = "knn", data = train)  # running the model without any tuning parameter

ggplot(train_knn, highlight = TRUE)

pred_knn <- predict(train_knn, test_x, type = "raw")
cm5 <- confusionMatrix(pred_knn, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "k-nearest neighbors (knn) - no tuning",
                                   Accuracy = cm5$overall["Accuracy"],
                                   Sensitivity = cm5$byClass[c("Sensitivity")],
                                   Specificity = cm5$byClass[c("Specificity")],
                                   Prevalence = cm5$byClass[c("Prevalence")],
                                   Kappa = cm5$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm5$table[1,1],
                                   FalsePositives = cm5$table[1,2],
                                   FalseNegatives = cm5$table[2,1],
                                   TrueNegatives = cm5$table[2,2]))
cm_results

varImp(train_knn)


# Now use various methods for model optimization


# Max accuracy

set.seed(1, sample.kind="Rounding")

# ks <- seq(3, 251, 2)
# ks <- seq(100, 300, 2)
ks <- seq(3, 300, 2)

accuracy <- map_df(ks, function(k){
  fit <- knn3(Disease ~ ., data = train, k = k)
  y_hat <- predict(fit, train, type = "class")
  cm_train <- confusionMatrix(data = y_hat, reference = train$Disease)
  train_error <- cm_train$overall["Accuracy"]
  y_hat <- predict(fit, test, type = "class")
  cm_test <- confusionMatrix(data = y_hat, reference = test$Disease)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})


#pick the k that maximizes accuracy
plot(ks,accuracy$test)
k1 <- ks[which.max(accuracy$test)]
max(accuracy$test)  


train_knn_maxacc <- knn3(Disease ~ ., data = train, k = k1)
pred_knn_maxacc <- predict(train_knn_maxacc, test, type = "class")

cm6 <- confusionMatrix(pred_knn_maxacc, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "k-nearest neighbors (knn) - max. acc.",
                                   Accuracy = cm6$overall["Accuracy"],
                                   Sensitivity = cm6$byClass[c("Sensitivity")],
                                   Specificity = cm6$byClass[c("Specificity")],
                                   Prevalence = cm6$byClass[c("Prevalence")],
                                   Kappa = cm6$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm6$table[1,1],
                                   FalsePositives = cm6$table[1,2],
                                   FalseNegatives = cm6$table[2,1],
                                   TrueNegatives = cm6$table[2,2]))
cm_results


# F1 score (balanced accuracy)

set.seed(1, sample.kind="Rounding")

F_1 <- sapply(ks, function(k){
  fit <- knn3(Disease ~ ., data = train, k = k)
  y_hat <- predict(fit, test, type = "class")
  
  F_meas(data = y_hat, reference = test$Disease)
})

plot(ks, F_1)
max(F_1)
k2 <- ks[which.max(F_1)]


train_knn_F1 <- knn3(Disease ~ ., data = train, k = k2)
pred_knn_F1 <- predict(train_knn_F1, test, type = "class")

cm7 <- confusionMatrix(pred_knn_F1, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "k-nearest neighbors (knn) - F1 score",
                                   Accuracy = cm7$overall["Accuracy"],
                                   Sensitivity = cm7$byClass[c("Sensitivity")],
                                   Specificity = cm7$byClass[c("Specificity")],
                                   Prevalence = cm7$byClass[c("Prevalence")],
                                   Kappa = cm7$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm7$table[1,1],
                                   FalsePositives = cm7$table[1,2],
                                   FalseNegatives = cm7$table[2,1],
                                   TrueNegatives = cm7$table[2,2]))
cm_results



# cross validation

set.seed(1, sample.kind="Rounding")

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(Disease ~ ., method = "knn", data = train,
                      tuneGrid = data.frame(k = seq(3, 50, 2)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)

train_knn_cv$bestTune


pred_knn_cv <- predict(train_knn_cv, test, type = "raw")

cm8 <- confusionMatrix(pred_knn_cv, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "k-nearest neighbors (knn) - cross valid.",
                                   Accuracy = cm8$overall["Accuracy"],
                                   Sensitivity = cm8$byClass[c("Sensitivity")],
                                   Specificity = cm8$byClass[c("Specificity")],
                                   Prevalence = cm8$byClass[c("Prevalence")],
                                   Kappa = cm8$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm8$table[1,1],
                                   FalsePositives = cm8$table[1,2],
                                   FalseNegatives = cm8$table[2,1],
                                   TrueNegatives = cm8$table[2,2]))
cm_results

varImp(train_knn_cv)


# Loess model ----

set.seed(1, sample.kind="Rounding")

grid <- expand.grid(span = seq(0.01, 0.8, len = 10), degree = 1)

train_loess <- train(Disease ~ ., method = "gamLoess", 
                     tuneGrid=grid,
                     data = train)

ggplot(train_loess, highlight = TRUE)

pred_loess <- predict(train_loess, test, type = "raw")

cm9 <- confusionMatrix(pred_loess, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Loess model",
                                   Accuracy = cm9$overall["Accuracy"],
                                   Sensitivity = cm9$byClass[c("Sensitivity")],
                                   Specificity = cm9$byClass[c("Specificity")],
                                   Prevalence = cm9$byClass[c("Prevalence")],
                                   Kappa = cm9$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm9$table[1,1],
                                   FalsePositives = cm9$table[1,2],
                                   FalseNegatives = cm9$table[2,1],
                                   TrueNegatives = cm9$table[2,2]))
cm_results

varImp(train_loess)


# Classification trees ----

set.seed(1, sample.kind="Rounding")

train_rpart <- train(Disease ~ ., method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.2, len = 25)),
                     data = train)

# train_rpart <- train(Disease ~ ., method = "rpart",
#                      # tuneGrid = data.frame(cp = seq(0.04, 0.2, len = 25)),
#                      control = rpart.control(minsplit=1, minbucket=1, cp=0.001),
#                      data = train)


plot(train_rpart)


rpart.plot(train_rpart$finalModel)


pred_rpart <- predict(train_rpart, test, type = "raw")

cm10 <- confusionMatrix(pred_rpart, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Classification trees",
                                   Accuracy = cm10$overall["Accuracy"],
                                   Sensitivity = cm10$byClass[c("Sensitivity")],
                                   Specificity = cm10$byClass[c("Specificity")],
                                   Prevalence = cm10$byClass[c("Prevalence")],
                                   Kappa = cm10$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm10$table[1,1],
                                   FalsePositives = cm10$table[1,2],
                                   FalseNegatives = cm10$table[2,1],
                                   TrueNegatives = cm10$table[2,2]))
cm_results


# extract predictor names

imp <- varImp(train_rpart)
imp

tree_terms <- as.character(unique(train_rpart$finalModel$frame$var[!(train_rpart$finalModel$frame$var == "<leaf>")]))
tree_terms

data_frame(term = rownames(imp$importance), 
           importance = imp$importance$Overall) %>%
  mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
  filter(term %in% tree_terms)





# Random forests ----

set.seed(1, sample.kind="Rounding")

train_rf <- train(train_x, train_y, method = "rf", 
                  tuneGrid = data.frame(mtry = seq(20, 200, 25)), importance = TRUE,
                  nodesize = 1)

ggplot(train_rf, highlight = TRUE)
train_rf$bestTune

pred_rf <- predict(train_rf, test_x, type = "raw")


cm11 <- confusionMatrix(pred_rf, test_y)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Random forests",
                                   Accuracy = cm11$overall["Accuracy"],
                                   Sensitivity = cm11$byClass[c("Sensitivity")],
                                   Specificity = cm11$byClass[c("Specificity")],
                                   Prevalence = cm11$byClass[c("Prevalence")],
                                   Kappa = cm11$overall[c("Kappa")],
                                   F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                   TruePositives = cm11$table[1,1],
                                   FalsePositives = cm11$table[1,2],
                                   FalseNegatives = cm11$table[2,1],
                                   TrueNegatives = cm11$table[2,2]))
cm_results

varImp(train_rf)


# Ensemble ----


# Creating an ensemble using the predictions from the most powerful models.
# The motivation behind creating an ensemble is that the combined power of several
# machine learning models is generally higher than the individual models.

# I have chosen the top 2 models with the highest F1 scores, because my areas of concern have been to keep
# False Negatives as low as possible while having high sensitivity.

cm_results %>% 
  arrange(desc(F1score)) %>% 
  slice(1:2)


ensemble <- cbind(
  # glm = pred_glm == "yes",
  # lda = pred_lda == "yes",
  # qda = pred_qda == "yes",
  knn = pred_knn_F1 == "yes",
  # loess = pred_loess == "yes",
  # rpart = pred_rpart == "yes",
  rf = pred_rf == "yes"
  # knn_cv = pred_knn_cv == "yes"
)

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "yes", "no")
acc_ensemble <- mean(ensemble_preds == test_y)
sens_ensemble <- sensitivity(factor(ensemble_preds), test_y, positive = "yes")
spec_ensemble <- specificity(factor(ensemble_preds), test_y, positive = "yes")
prev_ensemble <- posPredValue(factor(ensemble_preds), test_y, positive = "yes")
f1_score <- 2*(sens_ensemble*prev_ensemble)/(sens_ensemble+prev_ensemble)

cm_results <- bind_rows(cm_results, 
                        data_frame(method = "Ensemble",
                                   Accuracy = acc_ensemble,
                                   Sensitivity = sens_ensemble,
                                   Specificity = spec_ensemble,
                                   Prevalence = prev_ensemble,
                                   F1score = f1_score))
cm_results

cm_results %>% 
  arrange(desc(F1score))



# The ensemble seems to outperform the individual models, because its F1 is the highest.



# write.xlsx(cm_results,"cm_results_FINAL.xlsx")


# Final model ----

# The final step of the report is to apply the final model - the ensemble between KNN - F1 score,
# and Random forests, to the validation dataset.
# As a training set I am using the original training subset, after NAs and duplicates have been removed.
# My reason for not taking the training dataset after all data processing is because those steps were performed
# only to find the best models.


train_final <- liver_subset %>% 
  mutate(Gender = as.numeric(Gender == "Male")) %>% 
  mutate(Disease = as.factor(Disease))
train_final$Disease = factor(train_final$Disease, levels = c("yes","no"))


test_final <- validation %>% 
  mutate(Gender = as.numeric(Gender == "Male")) %>% 
  mutate(Disease = as.factor(Disease))
test_final$Disease = factor(test_final$Disease, levels = c("yes","no"))


# Remove rows with NA values and duplicates from the validation dataset.
test_final <- na.omit(test_final)
test_final <- unique(test_final)

# test_final_y <- factor(test_final$Disease, levels = c("yes", "no"))


# Apply the ensemble model ----


# # LDA
# 
# set.seed(1, sample.kind="Rounding")
# 
# train_lda <- train(as.matrix(train_final[,1:10]), as.factor(train_final$Disease), method = "lda")
# pred_lda <- predict(train_lda, as.matrix(test_final[,1:10]), type = "raw")
# cm_final_2 <- confusionMatrix(pred_lda, as.factor(test_final$Disease))
# 
# cm_final <- data_frame(method = "LDA",
#                        Accuracy = cm_final_2$overall["Accuracy"],
#                        Sensitivity = cm_final_2$byClass[c("Sensitivity")],
#                        Specificity = cm_final_2$byClass[c("Specificity")],
#                        Prevalence = cm_final_2$byClass[c("Prevalence")],
#                        F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
#                        TruePositives = cm_final_2$table[1,1],
#                        FalsePositives = cm_final_2$table[1,2],
#                        FalseNegatives = cm_final_2$table[2,1],
#                        TrueNegatives = cm_final_2$table[2,2])
# cm_final
# 
# varImp(train_lda)


# KNN - F1 score (balanced accuracy)

set.seed(1, sample.kind="Rounding")

train_knn_F1 <- knn3(Disease ~ ., data = train_final, k = 243)
pred_knn_F1 <- predict(train_knn_F1, test_final, type = "class")

cm_final_2 <- confusionMatrix(pred_knn_F1, as.factor(test_final$Disease))

cm_final <- data_frame(method = "KNN - F1",
                       Accuracy = cm_final_2$overall["Accuracy"],
                       Sensitivity = cm_final_2$byClass[c("Sensitivity")],
                       Specificity = cm_final_2$byClass[c("Specificity")],
                       Prevalence = cm_final_2$byClass[c("Prevalence")],
                       F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                       TruePositives = cm_final_2$table[1,1],
                       FalsePositives = cm_final_2$table[1,2],
                       FalseNegatives = cm_final_2$table[2,1],
                       TrueNegatives = cm_final_2$table[2,2])
cm_final


# Random forests

set.seed(1, sample.kind="Rounding")

train_rf <- train(as.matrix(train_final[,1:10]), as.factor(train_final$Disease), method = "rf", 
                  tuneGrid = data.frame(mtry = seq(20, 200, 25)), importance = TRUE,
                  nodesize = 1)

pred_rf <- predict(train_rf, as.matrix(test_final[,1:10]), type = "raw")

cm_final_3 <- confusionMatrix(pred_rf, as.factor(test_final$Disease))

cm_final <- bind_rows(cm_final, 
                      data_frame(method = "Random forests",
                                 Accuracy = cm_final_3$overall["Accuracy"],
                                 Sensitivity = cm_final_3$byClass[c("Sensitivity")],
                                 Specificity = cm_final_3$byClass[c("Specificity")],
                                 Prevalence = cm_final_3$byClass[c("Prevalence")],
                                 F1score = 2*(Prevalence*Sensitivity/(Prevalence+Sensitivity)),
                                 TruePositives = cm_final_3$table[1,1],
                                 FalsePositives = cm_final_3$table[1,2],
                                 FalseNegatives = cm_final_3$table[2,1],
                                 TrueNegatives = cm_final_3$table[2,2]))
cm_final

varImp(train_rf)


# Ensemble

ensemble <- cbind(
  # lda = pred_lda == "yes",
  knn = pred_knn_F1 == "yes",
  rf = pred_rf == "yes"
)

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "yes", "no")
acc_ensemble <- mean(ensemble_preds == as.factor(test_final$Disease))
sens_ensemble <- sensitivity(factor(ensemble_preds), as.factor(test_final$Disease), positive = "yes")
spec_ensemble <- specificity(factor(ensemble_preds), as.factor(test_final$Disease), positive = "yes")
prev_ensemble <- posPredValue(factor(ensemble_preds), as.factor(test_final$Disease), positive = "yes")
f1_score <- 2*(sens_ensemble*prev_ensemble)/(sens_ensemble+prev_ensemble)

cm_final <- bind_rows(cm_final, 
                      data_frame(method = "Ensemble",
                                 Accuracy = acc_ensemble,
                                 Sensitivity = sens_ensemble,
                                 Specificity = spec_ensemble,
                                 Prevalence = prev_ensemble,
                                 F1score = f1_score))
cm_final

cm_final %>% 
  arrange(desc(F1score))






# Results ----

# Applying the final model (the ensemble) to the validation dataset (the final hold-out test set),
# results in a good F1 score, however given the high prevalence of the data, I would not be comfortable enough
# to say that this is an optimal model, since it may out-predict true positives (sensitivity is very close to 1),
# as well as predict too many false positives (and run the risk of sending people home that actually have a liver
# disease):

cm_final[3,1:6]



# Conclusions ----

# Overall, this analysis has offered me many new insights into machine learning, and in particular into
# classification predictions. My knowledge as of now is still too limited and narrow to fully grasp all the
# possibilities and paths I could have taken in the analysis.

# For example, the entire report could have taken a different approach by using k-fold cross-validation
# on the entire dataset, instead of splitting it into train, test and validation sets.
# Second, I learned that there are many different ways to measure the predictive powers of any given model,
# and perhaps a good approach in general would be to know from the beginning what exactly you are trying to achieve.
# Another task is to pre-process data, which varies widely depending on the dataset. I have used a few methods,
# but there are many more out there which might have performed better.
# And last but not least are the machine learning algorithms themselves and tuning parameters, which I need to
# learn more about, as well look into algorithms that I have not used but could have potentially performed better.
# The measures to look at and optimize depend a lot on what the purpose of the analysis is.

# A big win in such an analysis would be, I suspect, to get insights from stakeholders
# on what they consider to be the most important parameters to optimize.
# If a panel of doctors would request such an analysis in real life,
# would they be most interested in keeping the False positives as low as possible,
# or would it be a combination of low False positives but also low False negatives,
# if let's say the hospital cannot afford misdiagnosis of healthy people (not enough availability of tests, no extra beds in hospitals etc.)

# In conclusion I would like to say thank you to professor Irizarry as well as the supporting teachers,
# for opening up my eyes to this vast expanse of new and exciting territory.
# My next step is to continue learning, and one day return to this report and try to make it better.

