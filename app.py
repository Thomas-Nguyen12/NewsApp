import dash 
import dash_html_components as html 
import dash_core_components as dcc 
from dash import Dash, html, dcc, callback, Output, Input
import joblib
# This module is for graphing 
import plotly.express as px
import pandas as pd
import plotly.express as px
import numpy as np
import sklearn.metrics.pairwise.cosine_similarity
# Requires Dash 2.17.0 or later
"""
app.layout() --> describes the layout and appearance of the app

I will use the 

"""

news_model = joblib.load("news_model.pkl")






# clean_df will contain the date and text 
clean_df = pd.read_csv("clean_df.csv") 


# extracted topics will contain the numerical count information for topics and dates 
extracted_topics = pd.read_csv("extracted_topics.csv") 

extracted_topics.drop(['Unnamed: 0'], axis=1, inplace=True)
unique_topics = extracted_topics['topic'].unique() 
tfidf_vectoriser = joblib.load('tfidf_vectoriser.pkl')


app = Dash()

# visualising the data (news trends over time, but the user chooses the topic)

# app.layout contains an array of divs that contain the interactive elements




app.layout = html.Div([
    html.H1(children='News Trends over time'),
    
    
    html.Div(children=[
        html.Label('Choose a news category'),
        dcc.Dropdown([*unique_topics], 'Law and crime', multi=False, id='topic-dropdown'),
        
        html.Br(),
        
        
        
        dcc.Graph(
            id='news-graph',
            figure={}
            
        ),

         
        
    ]),
    
    html.H1(children='News Classifier'),
    html.Div(children=[html.Label("This model will classifier will group news reports between 14 different topics")]),
    
    html.Div(children=[
        dcc.Textarea(
        id='text-box',
        value='Enter your news here',
        style={'width': '100%', 'height': 300}),
        
        
    ]),
    
    html.Div(id='classifier-output')    
])


# ---------------------
@callback(
    Output('news-graph', 'figure'),
    Input('topic-dropdown', 'value') 
)
def update_figure_topic(input_topic): 
    filtered_topic_df = extracted_topics[extracted_topics.topic == input_topic] 
    fig = px.line(filtered_topic_df, x='year_month', y='count', color='topic') 
    fig.update_layout(transition_duration=500) 
    return fig
    
# ---------------------- Implemeneting the model here
@callback(
    Output('classifier-output', 'children'),
    Input('text-box', 'value')
)

def classify_news(input_news):
    
    # vectorising the text
    input_news = [input_news]
    
    
    vectorised_text = tfidf_vectoriser.transform(input_news) 
    
    
    # passing the vectorised text into the model
    prediction = news_model.predict(vectorised_text).toarray()
    
    
    


    # finding out what the news category belonged to 
    prediction_df = pd.DataFrame(prediction, columns=extracted_topics.topic.unique())
    


    # returning the output
    
    # Maybe I can format the results so that only columns with a one in it are returned
    selected_cols = prediction_df.columns[prediction_df.iloc[0] == 1].tolist()
    
    
    return f"The prediction is: {selected_cols}..."


# This area will be for calculating similarity with known Ai models
def cosine_similarity(input_text: str) -> str:
    
    """
    (1) Pass the news report into the AI report
    (2) Assess similarity with AI models
    """
     pass


if __name__ == "__main__":
    app.run(debug=True, port=1234) 
    
    