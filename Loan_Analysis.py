#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[2]:


loan = pd.read_csv('C:\\Users\\hp\\Desktop\\Internship\\Dataset.csv',dtype = 'object')
print(loan.shape)


# In[3]:


loan.head()


# In[4]:


N_colm = loan.isnull().sum()
N_colm = N_colm[N_colm.values>(0.3*len(loan))]
plt.figure(figsize = (20,4))
N_colm.plot(kind = 'bar')
plt.title('List of Coloumn & NA counts where NA values are more than 30%')
plt.show()


# In[8]:


def removeNulls(dataframe,axis=1,percent=0.3):
    df = dataframe.copy()
    ishape = df.shape
    if axis == 0:
        rownames = df.transpose().idnull().sum()
        rownames = list(rownames[rownames.values > percent*len(df)].index)
        df.drop(df.index[rownames],inplace = True)
        print("\nNumber of Rows Dropped\t:",len(rownames))
    else:
        colnames = (df.isnull().sum()/len(df))
        colnames = list(colnames[colnames.values>=percent].index)
        df.drop(labels=colnames,axis=1,inplace = True)
        print("Number of Coloumns dropped\t:",len(colnames))
        print("/nOld dataset rows,coloumns",ishape,"/nNew dataset rows,coloumns",df.shape)
        return df


# In[9]:


loan = removeNulls(loan,axis=1,percent = 0.3)


# In[10]:


loan = removeNulls(loan,axis=1,percent = 0.3)


# In[11]:


unique = loan.nunique()
unique = unique[unique.values==1]


# In[12]:


loan.drop(labels = list(unique.index),axis=1,inplace=True)
print("Rows and Coloumns left",loan.shape)


# In[14]:


print(loan.emp_length.unique())
loan.emp_length.fillna('0',inplace = True)
loan.emp_length.replace(['n/a'],'Self-Employed',inplace = True)
print(loan.emp_length.unique())


# In[16]:


not_required_col = ["id","member_id","url","zip_code"]
loan.drop(labels = not_required_col,axis=1,inplace = True)
print("No. of Row and Columns left",loan.shape)


# In[20]:


import re
loan["int_rate"] = loan["int_rate"].str.extract(r'(\d+.+\d)')
loan_copy.head()


# In[21]:


numeric_col = ["loan_amnt","funded_amnt","funded_amnt_inv","installment","int_rate","annual_inc","dti"]
loan[numeric_col]=loan[numeric_col].apply(pd.to_numeric)


# In[22]:


loan.tail(3)


# In[23]:


(loan.purpose.value_counts()*100)/len(loan)


# In[26]:


del_loan_purpose = (loan.purpose.value_counts()*100)/len(loan)
del_loan_purpose =  del_loan_purpose[(del_loan_purpose <0.75)|(del_loan_purpose.index == 'other')]

loan.drop(labels = loan[loan.purpose.isin(del_loan_purpose.index)].index,inplace = True)
print("Rows and Coloumn Left",loan.shape)


print(loan.purpose.unique())


# In[27]:


(loan.loan_status.value_counts()*100)/len(loan)


# In[28]:


del_loan_status = (loan.loan_status.value_counts()*100)/len(loan)
del_loan_status = del_loan_status[(del_loan_status < 1.5)]

loan.drop(labels = loan[loan.loan_status.isin(del_loan_status.index)].index,inplace = True)

print("Rows and Coloumns left",loan.shape)
print(loan.loan_status.unique())


# In[29]:


loan['loan_income_ratio'] = loan['loan_amnt']/loan['annual_inc']


# In[32]:


loan['issue_month'],loan['issue_year'] = loan['issue_d'].str.split('-',1).str
loan[['issue_d','issue_month','issue_year']].head()


# In[33]:


month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
loan['Issue_month'] = pd.Categorical(loan['issue_month'],categories = month_order,ordered = True)


# In[35]:


bins = [0 , 5000 , 10000 , 15000 , 20000 , 25000 , 40000]
slot = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000 and above']
loan['loan_amnt_range'] = pd.cut(loan['loan_amnt'],bins,labels = slot)


# In[36]:


bins = [0,25000,50000,75000,100000,1000000]
slot = ['0-25000','25000-50000','50000-75000','75000-100000','100000 and above']
loan['annual_inc_range'] = pd.cut(loan['annual_inc'],bins,labels=slot)


# In[37]:


bins = [0,7.5,10,12.5,15,20]
slot = ['0-7.5','7.5-10','10-12.5','12.5-15','15 and above']
loan['int_rate_range'] = pd.cut(loan['int_rate'],bins,labels = slot)


# In[52]:


def univariate(df,col,vartype,hue=None):
    sns.set(style = "darkgrid")
    
    if vartype == 0:
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,8))
        ax[0].set_title("Distriution Plot")
        sns.distplot(df[col],ax = ax[0])
        ax[1].set_title("Violin Plot")
        sns.violinplot(data = df,x=col,ax = ax[1],inner = "quartile")
        ax[2].set_title("Box Plot")
        sns.boxplot(data = df , x=col,ax = ax[2],orient = 'v')
        
    if vartype == 1:
        temp = pd.Series(data = hue)
        fig , ax = plt.subplots()
        width = len(df[col].unique()) + 6 +4*len(temp.unique())
        fig.set_size_inches(width,7)
        ax = sns.countplot(data = df,x=col,order = df[col].value_counts().index,hue = hue)
        if len(temp.unique())>0:
            for p in ax.patches:
                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))),(p.get_x()+0.05,p.get_height()+20))
        else:
            for p in ax.patches:
                ax.annotate(p.get_height(),(p.get_x()+0.32,p.get_height()+20))
        del temp
        
    else:
        exit
        
    plt.show()
        


# In[45]:


univariate(df=loan,col='loan_amnt',vartype=0)


# In[46]:


univariate(df=loan,col='int_rate',vartype=0)


# In[47]:


loan["annual_inc"].describe()


# In[49]:


q = loan["annual_inc"].quantile(0.995)

loan = loan[loan["annual_inc"]<q]
loan["annual_inc"].describe()


# In[50]:


univariate(df=loan,col='annual_inc',vartype=0)


# In[53]:


univariate(df=loan,col='loan_status',vartype=1)


# In[54]:


univariate(df=loan,col='purpose',vartype=1,hue='loan_status')


# In[55]:


loan.home_ownership.unique()


# In[56]:


rem = ["OTHER","NONE","ANY"]
loan.drop(loan[loan['home_ownership'].isin(rem)].index,inplace = True)
loan.home_ownership.unique()


# In[57]:


univariate(df=loan,col='home_ownership',vartype=1,hue = 'loan_status')


# In[59]:


year_wise = loan.groupby(by=[loan.issue_year])[['loan_status']].count()
year_wise.rename(columns = {"loan_status":"count"},inplace = True)
ax = year_wise.plot(figsize = (20,8))
year_wise.plot(kind = 'bar',figsize= (20,8),ax = ax)
plt.show()


# In[60]:


univariate(df=loan,col='term',vartype=1,hue = 'loan_status')


# In[61]:


plt.figure(figsize = (16,12))
sns.boxplot(data = loan,x = 'purpose',y = 'loan_amnt',hue = 'loan_status')
plt.title('Purpose of loan vs Loan Amount')
plt.show()


# In[62]:


loan_correlation = loan.corr()
loan_correlation


# In[65]:


f,ax = plt.subplots(figsize = (14,9))
sns.heatmap(loan_correlation, xticklabels = loan_correlation.columns.values,yticklabels = loan_correlation.columns.values,annot = True)
plt.show()


# In[67]:


loanstatus = loan.pivot_table(index = ['loan_status','purpose','emp_length'],values = 'loan_amnt',aggfunc=('count')).reset_index()
loanstatus = loan.loc[loan['loan_status']=='Charged Off']


# In[68]:


ax = plt.figure(figsize = (30,18))
ax = sns.boxplot(x='emp_length',y = 'loan_amnt',hue = 'purpose',data = loanstatus)
ax.set_title('Employment Length vs Loan Amount for different purpose of loan',fontsize =22, weight = 'bold')
ax.set_xlabel('Employment Length',fontsize =16)
ax.set_ylabel('Loan Amount',color ='b',fontsize =16)
plt.show()


# In[76]:


def crosstab(df,col):
    crosstab = pd.crosstab(df[col],df['loan_status'],margins = True)
    crosstab['Probability_Charged_Off'] = round((crosstab['Charged Off']/crosstab['All']),3)
    crosstab = crosstab[0:-1]
    return crosstab


# In[80]:


def bivariate_prob(df,col,stacked = True):
    plotCrosstab = crosstab(df,col)
    
    linePlot = plotCrosstab[['Probability_Charged_Off']]
    barPlot = plotCrosstab.iloc[:,0:2]
    ax = linePlot.plot(figsize = (20,8),marker = 'o',color = 'b')
    ax2 = barPlot.plot(kind ='bar',ax = ax,rot=1,secondary_y =True,stacked = stacked)
    ax.set_title(df[col].name.title()+' vs Probability Charged Off',fontsize = 20,weight = 'bold')
    ax.set_xlabel(df[col].name.title(),fontsize =14)
    ax.set_ylabel('Probabilty Of Charged Off',color = 'b',fontsize ='14')
    ax2.set_ylabel('Number of Applicant',color = 'g',fontsize = 14)
    plt.show()


# In[81]:


filter_states  =loan.addr_state.value_counts()
filter_states = filter_states[(filter_states <10)]

loan_filter_states = loan.drop(labels = loan[loan.addr_state.isin(filter_states.index)].index)


# In[82]:


states = crosstab(loan_filter_states,'addr_state')
display(states.tail(20))

bivariate_prob(df = loan_filter_states,col = 'addr_state')


# In[83]:


purpose = crosstab(loan,'purpose')
display(purpose)

bivariate_prob(df=loan,col='purpose',stacked=False)


# In[84]:


grade = crosstab(loan,'grade')
display(grade)

bivariate_prob(df=loan,col='grade',stacked=False)
bivariate_prob(df=loan,col='sub_grade')


# In[85]:


annual_inc_range = crosstab(loan,'annual_inc_range')
display(annual_inc_range)

bivariate_prob(df=loan,col='annual_inc_range')


# In[86]:


int_rate_range = crosstab(loan,'int_rate_range')
display(int_rate_range)

bivariate_prob(df=loan,col='int_rate_range',stacked=False)


# In[88]:


emp_length = crosstab(loan,'emp_length')
display(emp_length)

bivariate_prob(df=loan,col='emp_length')


# In[ ]:




