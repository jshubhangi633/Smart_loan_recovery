# Smart loan recovery
## The dataset for this project contains critical attributes such as:
## 1. Demographic Information: Age, employment type, income level, and number of dependents.
## 2. Loan Details: Loan amount, tenure, interest rate, and collateral value.
## 3. Repayment History: Number of missed payments, days past due, and monthly EMI payments.
## 4. Collection Efforts: Collection methods used, number of recovery attempts, and legal actions taken.
## 5. Loan Recovery Status: Whether the loan was fully recovered, partially recovered, or remains outstanding.

 ### [Link to the dataset](https://amanxai.com/wp-content/uploads/2025/02/loan-recovery.csv)


### Importing the dataset
```py
import pandas as pd
df = pd.read_csv("/content/loan recovery.csv")
print(df.head())
```
### Let’s have a look at the summary statistics of the data
```py
df.describe()
```
### Analyzing Data Distribution and Relationships
```py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


fig = px.histogram(df, x='Loan_Amount', nbins=30, marginal="violin", opacity=0.7,
                   title="Loan Amount Distribution & Relationship with Monthly Income",
                   labels={'Loan_Amount': "Loan Amount (in $)", 'Monthly_Income': "Monthly Income"},
                   color_discrete_sequence=["royalblue"])

fig.add_trace(go.Scatter(
    x=sorted(df['Loan_Amount']),
    y=px.histogram(df, x='Loan_Amount', nbins=30, histnorm='probability density').data[0]['y'],
    mode='lines',
    name='Density Curve',
    line=dict(color='red', width=2)
))

scatter = px.scatter(df, x='Loan_Amount', y='Monthly_Income',
                     color='Loan_Amount', color_continuous_scale='Viridis',
                     size=df['Loan_Amount'], hover_name=df.index)

for trace in scatter.data:
    fig.add_trace(trace)

fig.update_layout(
    annotations=[
        dict(
            x=max(df['Loan_Amount']) * 0.8, y=max(df['Monthly_Income']),
            text="Higher Loan Amounts are linked to Higher Income Levels",
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color="red")
        )
    ],
    xaxis_title="Loan Amount (in $)",
    yaxis_title="Monthly Income (in $)",
    template="plotly_white",
    showlegend=True
)

fig.show()
```

