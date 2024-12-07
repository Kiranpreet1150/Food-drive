import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import streamlit.components.v1 as components


# Load the dataset with a specified encoding
data = pd.read_csv('data_2024.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('logo.jpeg', use_container_width=True)

    st.subheader("💡 Abstract:")

    inspiration = '''
    The Edmonton Food Drive Project aims to automate route assignment and improve pick-up efficiency based on area and donation volume. Use insights from the model to predict future donation patterns and refine strategies for continuous improvement. Enhance communication and coordination between Regional Coordinators, Stake Food Drive Representatives, and Ward Food Drive Representatives to streamline operations.
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    The Edmonton City Food Drive project focuses on using machine learning to optimize food donation management in Edmonton by analyzing the data collected in 2023 and 2024. It aims to improve drop-off and pick-up efficiency, enhance route planning, and optimize resource allocation for a more effective food drive campaign.
    '''

    st.write(what_it_does)






def clustermap():
    st.title('Streamlit App with Embedded My Google Map')
    st.write('The map with clusters of neighbourhood: ')

    # Read the HTML file contents
    with open('cluster_map.html', 'r') as f:
        html_string = f.read()

    # Display the HTML using components.html
    components.html(html_string, width=600, height=450, scrolling=True)

def Visualizations():
    st.title('Visualizations')
    st.write('Visualizations for the Edmonton Food drive 2024')

    # Embedding Google Map using HTML iframe
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/f776dd4d-6716-486a-bb42-47b56af67bf2/page/z0RYE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    neighbourhood_option = data['Neighbourhood'].unique().tolist()
    neighbourhood = st.selectbox("Neighbourhood", neighbourhood_option)
    stake_option = data["Stake"].unique().tolist()
    stake = st.selectbox("Stake", stake_option)
    route_option = data['New Route Number/Name'].unique().tolist()
    route = st.selectbox("New Route Number/Name", route_option)
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent", 10, 300, 60)
    doors_in_route = st.slider("Doors in Route", 10, 500, 100)

    # Predict button
    if st.button("Predict"):
      try:
        # Load the trained model
        model = joblib.load('model.pkl')

        # Prepare input data
        input_data = [[
            neighbourhood, route,
            routes_completed, doors_in_route, time_spent, stake
        ]]

        # Create a DataFrame with correct column names
        input_df = pd.DataFrame(input_data, columns=[
            'Neighbourhood', 'New Route Number/Name','Routes Completed', 'Doors in Route',
            'Time Spent', 'Stake'
        ])

        # Transform input data using the model's preprocessor
        preprocessor = model.named_steps['preprocessor']
        input_data_transformed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.named_steps['regressor'].predict(input_data_transformed)

        # Display prediction
        st.success(f"Predicted Donation Bags: {prediction[0]:.2f}")

      except ValueError as e:
        st.error(f"Input error: {e}")
      except Exception as e:
        st.error(f"An error occurred: {e}")




# Page 4: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"
    st.markdown(f"[Fill out the form]({google_form_url})")


# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "Maps", "ML Modeling", "Data Collection"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        Visualizations()
    elif app_page == "Maps":
        clustermap()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Data Collection":
        data_collection()


if __name__ == "__main__":
    main()
