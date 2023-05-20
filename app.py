import streamlit as st
import holisticai
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from holisticai.datasets import load_law_school
from holisticai.bias.plots import group_pie_plot, distribution_plot
from holisticai.bias.mitigation import Reweighing

st.markdown("# Classification with Machine Learning")
st.sidebar.write(
    """This is a dashboard to measure and mitigate bias with HolisticAI library."""
)

# Dataframe
df = load_law_school()['frame']

step1, step2, step3, step4 = st.tabs(["Step 1: Data Description", 
                                      "Step 2: Training Model", 
                                      "Step 3: Bias Metrics", 
                                      "Step 4: Bias Mitigation"])

with step1:
    st.subheader("Descriptive Analysis")
    
    st.write('The group_pie_plot function is called to create a pie plot of the protected attribute, using the p_attr variable, and the resulting plot is displayed in the first subplot using the ax parameter.')
    st.write('The distribution_plot function is called to create a distribution plot of the *ugpagt3* column, with respect to the *gender* column, and the resulting plot is displayed in the second subplot using the ax parameter.')
    p_attr = df['race1'] # protected attribute (race)
    y = df['bar']        # binary label vector

    fig1, ax = plt.subplots(1, 2, figsize=(10,3))
    # create a pie plot with protected attribute 
    group_pie_plot(p_attr, ax=ax[0]) 

    # create a distplot with target and gender variables
    distribution_plot(df['ugpagt3'], df['gender'], ax=ax[1])
    plt.tight_layout()

    st.write("")
    st.pyplot(fig1)

with step2:
    # machine learning models 
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    
    models = [lr, rf]
    
    # model selector
    model = st.selectbox("Select a Model", models)
    
    # simple preprocessing before training.
    df_enc = df.copy()
    df_enc['bar'] = df_enc['bar'].replace({'FALSE':0, 'TRUE':1})

    # split features and target, then train test split
    X = df_enc.drop(columns=['bar', 'ugpagt3'])
    y = df_enc['bar']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
    
    # StandarScaler transformation
    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train.drop(columns=['race1', 'gender']))
    X_test_t = scaler.transform(X_test.drop(columns=['race1', 'gender']))

    # fit model
    model.fit(X_train_t, y_train)

    # predictions
    y_pred = model.predict(X_test_t)

    # create image ROC Curve
    disp = RocCurveDisplay.from_estimator(model, X_test_t, y_test)

    # plot ROC Curve
    fig2, ax = plt.subplots(figsize=(5,3))
    disp.plot(ax=ax)
    plt.tight_layout()

    st.write("")
    st.pyplot(fig2, clear_figure=True, use_container_width=True)

with step3:

    # set up groups, prediction array and true (aka target/label) array.
    group_a = X_test["race1"]=='non-white'  # non-white vector
    group_b = X_test["race1"]=='white'      # white vector
    y_true  = y_test                        # true vector

    bias_metrics = holisticai.bias.metrics.classification_bias_metrics(group_a, group_b, y_pred, y_test, metric_type='both')

    # generate table with bias metrics
    st.table(bias_metrics)

with step4:

    reweighing_mitigator = Reweighing()

    reweighing_mitigator.fit(y_train, group_a, group_b)

    # access the new sample_weight
    sw = reweighing_mitigator.estimator_params["sample_weight"]

    model.fit(X_train_t, y_train, sample_weight=sw)

    # predictions
    y_pred = model.predict(X_test_t)

    bias_metrics_mitigated = holisticai.bias.metrics.classification_bias_metrics(group_a, group_b, y_pred, y_test, metric_type='both')

    # generate table with bias metrics
    st.table(bias_metrics_mitigated)

