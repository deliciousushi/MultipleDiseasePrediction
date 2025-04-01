import streamlit as st

# Get doctor name from URL parameters
query_params = st.experimental_get_query_params()
doctor_name = query_params.get("doctor", [""])[0]

st.title("Enter Patient Details")

if doctor_name:
    st.subheader(f"Appointment with {doctor_name}")

    # Patient details form
    with st.form("patient_details_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        contact = st.text_input("Contact Number")
        symptoms = st.text_area("Describe Symptoms")

        # Submit button
        submitted = st.form_submit_button("Confirm Appointment")

        if submitted:
            st.success(f"✅ Appointment confirmed for {name} with {doctor_name}.")
else:
    st.warning("No doctor selected. Please go back and book an appointment.")
