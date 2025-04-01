import streamlit as st
import time  # For adding delay after submission

# Get doctor name from URL parameters
query_params = st.experimental_set_query_params()
doctor_name = query_params.get("doctor", "")

st.title("Enter Patient Details")

if doctor_name:
    st.subheader(f"Appointment with Dr. {doctor_name}")

    # Patient details form
    with st.form("patient_details_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        contact = st.text_input("Contact Number")
        symptoms = st.text_area("Describe Symptoms")

        # Submit button
        submitted = st.form_submit_button("Confirm Appointment")

        if submitted:
            if name and contact:
                st.success(f"✅ Appointment confirmed for {name} with Dr. {doctor_name}.")
                
                # Simulate processing time
                time.sleep(2)

                # Redirect (optional: change the URL for further actions)
                st.query_params.clear()
                st.rerun()
            else:
                st.error("⚠️ Please fill in all required fields.")
else:
    st.warning("⚠️ No doctor selected. Please go back and book an appointment.")
