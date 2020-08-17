#!/usr/bin/env python
# coding: utf-8

# In[28]:


from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[30]:


true_rate_control = 0.69

true_rate_experiment = 0.7

people_visting_site = 500

# Create a numpy array with a shape of 2 by number of people visiting the site
# the first row represents the control outcomes and the second row is the experimental outcomes

control, experiment = np.random.rand(2, people_visting_site)

#if the random number is above than the true rate it is counted as a success, else failure 
control_successes = sum(control < true_rate_control)
experiment_successes = sum(experiment < true_rate_experiment)

control_failures = people_visting_site - control_successes
experiment_failures = people_visting_site - experiment_successes


# In[31]:


# Our Priors
prior_successes = 1
prior_failures = 1
prior_distribution = beta(prior_successes, prior_failures)

#graph of prior distribution
fig, ax = plt.subplots()

x_ = np.linspace(0, 1, 1000)
ax.plot(x_, prior_distribution.pdf(x_))

print(f'Successes: {prior_successes}')
print(f'Failures: {prior_failures}')


# In[32]:


# For our graph
fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 

# Control
control_alpha = control_successes + prior_successes
control_beta = control_failures + prior_failures

# Experiment
experiment_alpha = experiment_successes + prior_successes
experiment_beta = experiment_failures + prior_failures


# Generate beta distributions based on number of successes(alpha) and failures(beta)
control_distribution = beta(control_alpha, control_beta)
experiment_distribution = beta(experiment_alpha, experiment_beta)

#plot distributions using 

x = np.linspace(0, 1, 1000)
ax.plot(x, control_distribution.pdf(x))
ax.plot(x, experiment_distribution.pdf(x))

ax.set(xlabel='conversion rate', ylabel='density');

print(f'control_successes: {control_successes}')
print(f'control_failures: {control_failures}')
print('--------------------------')
print(f'experiment_successes: {experiment_successes}')
print(f'experiment_failures: {experiment_failures}')


# In[33]:


sample_size = people_visting_site*2
c_samples = pd.Series([control_distribution.rvs() for _ in range(sample_size)])
e_samples = pd.Series([experiment_distribution.rvs() for _ in range(sample_size)])

p_ish_value = 1.0 - sum(e_samples > c_samples)/sample_size
p_ish_value


# In[34]:


additional_visitors = 10000

# Control is Alpaca, Experiment is Bear
control, experiment = np.random.rand(2, additional_visitors)

# Add to existing data
control_successes += sum(control < true_rate_control)
experiment_successes += sum(experiment < true_rate_experiment)

fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 

control_failures += additional_visitors - sum(control < true_rate_control)
experiment_failures += additional_visitors - sum(experiment < true_rate_experiment)

x = np.linspace(0, 1, 1000)
ax.plot(x, control_distribution.pdf(x))
ax.plot(x, experiment_distribution.pdf(x))

ax.set(xlabel='conversion rate', ylabel='density');

print(f'control_successes: {control_successes}')
print(f'control_failures: {control_failures}')
print('--------------------------')
print(f'experiment_successes: {experiment_successes}')
print(f'experiment_failures: {experiment_failures}')


# In[15]:


data_total.head()


# In[19]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


# In[21]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()


# In[27]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, 
                feature_names=X_train_refined.columns,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[23]:


X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_test_refined)


# In[26]:





# In[ ]:




