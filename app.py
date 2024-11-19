from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load datasets
tourism_rating = pd.read_csv('data/tourism_rating.csv')
tourism_with_id = pd.read_csv('data/tourism_with_id.csv')

# Preprocessing
def preprocess_data():
    # Merge tourism_rating with tourism_with_id to get place_name
    merged_data = pd.merge(tourism_rating, tourism_with_id, on='place_id')
    
    # Group by user_id and place_id to calculate average ratings
    grouped_data = merged_data.groupby(['user_id', 'place_id'], as_index=False)['place_rating'].mean()
    
    # Calculate the overall average rating per place
    place_avg_rating = grouped_data.groupby('place_id', as_index=False)['place_rating'].mean()
    place_avg_rating = place_avg_rating.rename(columns={'place_rating': 'avg_rating'})
    
    # Join with tourism_with_id to add place_name
    place_avg_rating = pd.merge(place_avg_rating, tourism_with_id[['place_id', 'place_name']], on='place_id')
    
    return grouped_data, place_avg_rating

grouped_data, place_avg_rating = preprocess_data()

# Split data into train and test sets (80-20 split)
train_data, test_data = train_test_split(grouped_data, test_size=0.2, random_state=42)

# Flask route to get recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    
    # Filter training data for the given user_id
    user_data = train_data[train_data['user_id'] == user_id]
    
    if user_data.empty:
        return jsonify({"error": "No data found for this user"}), 404
    
    # Get the place_id with the highest average rating
    user_avg_rating = user_data.groupby('place_id')['place_rating'].mean().reset_index()
    top_place_id = user_avg_rating.loc[user_avg_rating['place_rating'].idxmax(), 'place_id']
    
    # Get the place_name for the top_place_id
    top_place = place_avg_rating[place_avg_rating['place_id'] == top_place_id]
    
    if top_place.empty:
        return jsonify({"error": "No place found for the given user data"}), 404
    
    top_place_name = top_place.iloc[0]['place_name']
    
    return jsonify({
        "user_id": user_id,
        "recommended_place": top_place_name
    })

if __name__ == '__main__':
    app.run(debug=True)
