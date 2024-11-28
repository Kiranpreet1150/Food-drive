import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the dataset with a specified encoding
data = pd.read_csv('data_2024.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('logo.jpeg', use_column_width=True)

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


# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Rename columns for clarity
    data_cleaned = data.rename(columns={
        'Neighbourhood': 'Neighbourhood',
        'Stake': 'Stake',
        'Ward/Branch': 'Ward/Branch',
        'New Route Number/Name': 'Route',
        '# of Adult Volunteers': '# of Adult Volunteers',
        '# of Youth Volunteers': '# of Youth Volunteers',
        'Donation Bags Collected': 'Donation Bags Collected',
        'Time Spent': 'Time to Complete (min)',
        'Routes Completed': 'Routes Completed',
        'Doors in Route': 'Doors in Route',
        'Assessed Value': 'Assessed Value',
    })

    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='# of Adult Volunteers', nbins=20, labels={'# of Adult Volunteers': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='# of Youth Volunteers', nbins=20, labels={'# of Youth Volunteers': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Donation Bags Collected', nbins=20, labels={'Donation Bags Collected': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Time to Complete (min)', nbins=20, labels={'Time to Complete (min)': 'Time to Complete'})
    st.plotly_chart(fig)


# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    neighbourhood = st.text_input("Neighbourhood")
    stake = st.text_input("Stake")
    ward_branch = st.text_input("Ward/Branch")
    route = st.text_input("Route")
    assessed_value = st.text_input("Assessed Value")
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10)

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('best_model_2.pkl')

        # Prepare input data for prediction
        input_data = [[ neighbourhood, route, assessed_value, adult_volunteers, youth_volunteers, routes_completed, doors_in_route, time_spent,ward_branch, stake]]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed


# Page 4: Neighbourhood Mapping
geodata = pd.read_csv("Food_Drive_2024.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")


# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"
    st.markdown(f"[Fill out the form]({google_form_url})")


# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Neighbourhood Mapping", "Data Collection"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()


if __name__ == "__main__":
    main()
