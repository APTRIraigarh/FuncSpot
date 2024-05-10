from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__,static_url_path='/static')

# Load data from Excel
data = pd.read_excel('data/nppd.xlsx')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'].values.astype('U'))

# Function to find matching functions
def get_top_matches(query, tfidf_matrix, n=6):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-n-1:-1]
    return related_docs_indices

def find_matching_functions(user_query, data, tfidf_matrix, n=6):
    top_matches_indices = get_top_matches(user_query, tfidf_matrix, n)
    return data.iloc[top_matches_indices]

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['query']
    results = find_matching_functions(user_query, data, tfidf_matrix)
    return render_template('results.html', results=results)

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        # Load data from Excel
        data = pd.read_excel('data/contribution_sheet.xlsx')
        
        # Get data from the form
        function_name = request.form['function_name']
        description = request.form['description']
        examples = request.form['examples']
        
        # Append data to the Excel file
        new_data = pd.DataFrame({'function_name': [function_name], 'description': [description], 'examples': [examples]})
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_excel('data/contribution_sheet.xlsx', index=False)
        
        return render_template('contribution_success.html')
    return render_template('contribute.html')


if __name__ == '__main__':
    app.run(debug=True)
